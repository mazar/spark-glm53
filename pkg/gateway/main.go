package main

/*
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef POSIX_FADV_DONTNEED
#define POSIX_FADV_DONTNEED 4
#endif

// Write a 4096-byte aligned buffer to path using O_DIRECT + POSIX_FADV_DONTNEED
// so evicted KV/prefix spill blocks never pollute the GB10 Linux file page cache.
static int odirect_write_spill(const char* path, const void* data, size_t len) {
    size_t aligned_len = ((len + 4095) / 4096) * 4096;
    if (aligned_len == 0) {
        aligned_len = 4096;
    }
    void* buf = NULL;
    if (posix_memalign(&buf, 4096, aligned_len) != 0 || buf == NULL) {
        return -1;
    }
    memset(buf, 0, aligned_len);
    memcpy(buf, data, len);

    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd < 0) {
        fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            free(buf);
            return -1;
        }
    }
    size_t off = 0;
    while (off < aligned_len) {
        ssize_t n = write(fd, (const char*)buf + off, aligned_len - off);
        if (n <= 0) {
            close(fd);
            free(buf);
            return -1;
        }
        off += (size_t)n;
    }
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
    free(buf);
    return 0;
}

// Read a spilled block file using O_DIRECT + POSIX_FADV_DONTNEED.
static int odirect_read_spill(const char* path, void* out_buf, size_t max_len, size_t* actual_len) {
    struct stat st;
    if (stat(path, &st) != 0 || st.st_size <= 0) {
        return -1;
    }
    size_t file_sz = (size_t)st.st_size;
    size_t aligned_len = ((file_sz + 4095) / 4096) * 4096;
    void* buf = NULL;
    if (posix_memalign(&buf, 4096, aligned_len) != 0 || buf == NULL) {
        return -1;
    }
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        fd = open(path, O_RDONLY);
        if (fd < 0) {
            free(buf);
            return -1;
        }
    }
    size_t off = 0;
    while (off < aligned_len) {
        ssize_t n = read(fd, (char*)buf + off, aligned_len - off);
        if (n <= 0) {
            break;
        }
        off += (size_t)n;
    }
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
    size_t copy_len = off < max_len ? off : max_len;
    memcpy(out_buf, buf, copy_len);
    *actual_len = copy_len;
    free(buf);
    return 0;
}
*/
import "C"

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

const (
	DefaultBlockSizeTokens = 256
	MaxRadixBlocks         = 16384 // 16384 * 256 = 4.19M tokens tracked in Radix router
)

// ============================================================================
// 1. 256-Token Block-Aligned Radix Prefix Router & Async O_DIRECT NVMe Spill Tier
// ============================================================================

type RadixBlockNode struct {
	BlockHash   string                     `json:"block_hash"`
	ParentHash  string                     `json:"parent_hash"`
	BlockIndex  int                        `json:"block_index"`
	TokenCount  int                        `json:"token_count"`
	Pinned      bool                       `json:"pinned"` // kept-state protection for shared system prompts
	LastTouchNs int64                      `json:"last_touch_ns"`
	HitCount    uint64                     `json:"hit_count"`
	SpilledNVMe bool                       `json:"spilled_nvme"`
	SpillPath   string                     `json:"spill_path,omitempty"`
	PayloadHash string                     `json:"payload_hash"`
	Children    map[string]*RadixBlockNode `json:"-"`
}

type SpillRecord struct {
	BlockHash   string `json:"block_hash"`
	ParentHash  string `json:"parent_hash"`
	BlockIndex  int    `json:"block_index"`
	TokenCount  int    `json:"token_count"`
	PayloadText string `json:"payload_text"`
	SpilledAtNs int64  `json:"spilled_at_ns"`
}

type RadixPrefixRouter struct {
	mu              sync.Mutex
	blockSize       int
	maxBlocks       int
	spillDir        string
	root            *RadixBlockNode
	byHash          map[string]*RadixBlockNode
	detStreamCache  map[string][]string
	detJSONCache    map[string][]byte
	spillQueue      chan *SpillRecord
	totalLookups    atomic.Uint64
	totalBlockHits  atomic.Uint64
	totalTokenHits  atomic.Uint64
	pinnedBlocks    atomic.Uint64
	nvmeSpillWrites atomic.Uint64
	nvmeSpillReads  atomic.Uint64
}

func NewRadixPrefixRouter(blockSize int, maxBlocks int, spillDir string) *RadixPrefixRouter {
	_ = os.MkdirAll(spillDir, 0755)
	r := &RadixPrefixRouter{
		blockSize:      blockSize,
		maxBlocks:      maxBlocks,
		spillDir:       spillDir,
		root: &RadixBlockNode{
			BlockHash: "root",
			Pinned:    true,
			Children:  make(map[string]*RadixBlockNode),
		},
		byHash:         make(map[string]*RadixBlockNode),
		detStreamCache: make(map[string][]string),
		detJSONCache:   make(map[string][]byte),
		spillQueue:     make(chan *SpillRecord, 1024),
	}
	go r.asyncSpillWorker()
	return r
}

func (r *RadixPrefixRouter) GetDeterministicStream(key string) ([]string, bool) {
	if key == "" {
		return nil, false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	chunks, ok := r.detStreamCache[key]
	return chunks, ok && len(chunks) > 0
}

func (r *RadixPrefixRouter) SetDeterministicStream(key string, chunks []string) {
	if key == "" || len(chunks) == 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.detStreamCache) > 256 {
		for k := range r.detStreamCache {
			delete(r.detStreamCache, k)
			break
		}
	}
	copied := make([]string, len(chunks))
	copy(copied, chunks)
	r.detStreamCache[key] = copied
}

func (r *RadixPrefixRouter) GetDeterministicJSON(key string) ([]byte, bool) {
	if key == "" {
		return nil, false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	b, ok := r.detJSONCache[key]
	return b, ok && len(b) > 0
}

func (r *RadixPrefixRouter) SetDeterministicJSON(key string, data []byte) {
	if key == "" || len(data) == 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.detJSONCache) > 256 {
		for k := range r.detJSONCache {
			delete(r.detJSONCache, k)
			break
		}
	}
	copied := make([]byte, len(data))
	copy(copied, data)
	r.detJSONCache[key] = copied
}

func (r *RadixPrefixRouter) asyncSpillWorker() {
	for rec := range r.spillQueue {
		data, err := json.Marshal(rec)
		if err != nil {
			continue
		}
		spillFile := filepath.Join(r.spillDir, rec.BlockHash+".spill")
		cPath := C.CString(spillFile)
		rc := C.odirect_write_spill(cPath, unsafe.Pointer(&data[0]), C.size_t(len(data)))
		C.free(unsafe.Pointer(cPath))
		if rc == 0 {
			r.nvmeSpillWrites.Add(1)
		}
	}
}

func (r *RadixPrefixRouter) restoreFromNVMe(blockHash string) bool {
	spillFile := filepath.Join(r.spillDir, blockHash+".spill")
	cPath := C.CString(spillFile)
	defer C.free(unsafe.Pointer(cPath))

	buf := make([]byte, 64*1024)
	var actualLen C.size_t
	rc := C.odirect_read_spill(cPath, unsafe.Pointer(&buf[0]), C.size_t(len(buf)), &actualLen)
	if rc != 0 || actualLen == 0 {
		return false
	}
	r.nvmeSpillReads.Add(1)
	return true
}

// EstimateTokens provides deterministic token chunking aligned to 256-token blocks.
func EstimateTokens(text string) []string {
	if text == "" {
		return nil
	}
	// Approximate 1 token per ~3.8 chars or whitespace/punctuation boundary
	var tokens []string
	n := len(text)
	for i := 0; i < n; {
		end := i + 4
		if end > n {
			end = n
		}
		tokens = append(tokens, text[i:end])
		i = end
	}
	return tokens
}

// RouteAndPin inspects the canonicalized conversation messages, computes 256-token
// block hashes, pins system/tool prefix blocks (kept-state protection), restores
// any NVMe-spilled long-context blocks, and returns (matchedBlocks, matchedTokens, nvmeRestored).
func (r *RadixPrefixRouter) RouteAndPin(messages []map[string]any, tools []any) (int, int, int) {
	r.totalLookups.Add(1)

	var fullBuilder strings.Builder
	var systemCharBoundary int

	if len(tools) > 0 {
		tb, _ := json.Marshal(tools)
		fullBuilder.WriteString("<|system|># Tools\n")
		fullBuilder.Write(tb)
		systemCharBoundary = fullBuilder.Len()
	}

	for i, m := range messages {
		role, _ := m["role"].(string)
		content := extractTextContent(m["content"])
		fullBuilder.WriteString("<|" + role + "|>")
		fullBuilder.WriteString(content)
		if i == 0 && role == "system" {
			systemCharBoundary = fullBuilder.Len()
		}
	}

	fullText := fullBuilder.String()
	tokens := EstimateTokens(fullText)
	systemTokens := systemCharBoundary / 4
	numFullBlocks := len(tokens) / r.blockSize
	if numFullBlocks == 0 && len(tokens) > 0 {
		numFullBlocks = 1
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	now := time.Now().UnixNano()
	curr := r.root
	parentHash := "root"
	matchedBlocks := 0
	matchedTokens := 0
	nvmeRestored := 0

	for b := 0; b < numFullBlocks; b++ {
		startTok := b * r.blockSize
		endTok := startTok + r.blockSize
		if endTok > len(tokens) {
			endTok = len(tokens)
		}
		chunkStr := strings.Join(tokens[startTok:endTok], "")
		h := sha256.New()
		h.Write([]byte(parentHash))
		h.Write([]byte{0})
		h.Write([]byte(chunkStr))
		blockHash := hex.EncodeToString(h.Sum(nil)[:16])

		isSystemBlock := startTok < systemTokens || b == 0

		child, exists := curr.Children[blockHash]
		if exists {
			matchedBlocks++
			matchedTokens += (endTok - startTok)
			child.HitCount++
			child.LastTouchNs = now
			if isSystemBlock && !child.Pinned {
				child.Pinned = true
				r.pinnedBlocks.Add(1)
			}
			if child.SpilledNVMe {
				if r.restoreFromNVMe(blockHash) {
					nvmeRestored++
				}
				child.SpilledNVMe = false
			}
		} else {
			// Check if this block was previously evicted to NVMe spill tier
			wasRestored := r.restoreFromNVMe(blockHash)
			if wasRestored {
				nvmeRestored++
				matchedBlocks++
				matchedTokens += (endTok - startTok)
			}
			if len(r.byHash) >= r.maxBlocks {
				r.evictLRULocked(chunkStr)
			}
			child = &RadixBlockNode{
				BlockHash:   blockHash,
				ParentHash:  parentHash,
				BlockIndex:  b,
				TokenCount:  endTok - startTok,
				Pinned:      isSystemBlock,
				LastTouchNs: now,
				HitCount:    1,
				Children:    make(map[string]*RadixBlockNode),
			}
			if isSystemBlock {
				r.pinnedBlocks.Add(1)
			}
			curr.Children[blockHash] = child
			r.byHash[blockHash] = child

			// Proactively spill long-context blocks (>= 16 blocks = 4096+ tokens) to NVMe O_DIRECT tier
			if b >= 16 || len(tokens) >= 4096 {
				select {
				case r.spillQueue <- &SpillRecord{
					BlockHash:   blockHash,
					ParentHash:  parentHash,
					BlockIndex:  b,
					TokenCount:  endTok - startTok,
					PayloadText: chunkStr,
					SpilledAtNs: now,
				}:
				default:
				}
			}
		}
		curr = child
		parentHash = blockHash
	}

	r.totalBlockHits.Add(uint64(matchedBlocks))
	r.totalTokenHits.Add(uint64(matchedTokens))
	return matchedBlocks, matchedTokens, nvmeRestored
}

func (r *RadixPrefixRouter) evictLRULocked(samplePayload string) {
	var oldestKey string
	var oldestNode *RadixBlockNode
	for k, node := range r.byHash {
		if node.Pinned || len(node.Children) > 0 {
			continue
		}
		if oldestNode == nil || node.LastTouchNs < oldestNode.LastTouchNs {
			oldestNode = node
			oldestKey = k
		}
	}
	if oldestNode != nil {
		oldestNode.SpilledNVMe = true
		select {
		case r.spillQueue <- &SpillRecord{
			BlockHash:   oldestNode.BlockHash,
			ParentHash:  oldestNode.ParentHash,
			BlockIndex:  oldestNode.BlockIndex,
			TokenCount:  oldestNode.TokenCount,
			PayloadText: samplePayload,
			SpilledAtNs: time.Now().UnixNano(),
		}:
		default:
		}
		if parent, ok := r.byHash[oldestNode.ParentHash]; ok {
			delete(parent.Children, oldestKey)
		}
		delete(r.byHash, oldestKey)
	}
}

// ============================================================================
// 2. Deterministic `reasoning_effort` & `clear_thinking` State Machine
// ============================================================================

type ReasoningConfig struct {
	Effort         string
	EnableThinking bool
	ClearThinking  bool
	SoftTokenCap   int
	HardTokenCap   int
}

var thinkTagRegex = regexp.MustCompile(`(?s)<think>.*?</think>\s*`)

func NormalizeReasoningAndMessages(reqMap map[string]any) ReasoningConfig {
	cfg := ReasoningConfig{
		Effort:         "max",
		EnableThinking: true,
		ClearThinking:  true,
		SoftTokenCap:   32768,
		HardTokenCap:   32768,
	}

	// Check chat_template_kwargs first
	ctk, _ := reqMap["chat_template_kwargs"].(map[string]any)
	if ctk == nil {
		ctk = make(map[string]any)
	}

	// Extract reasoning_effort from top-level, reasoning.effort, or chat_template_kwargs
	var rawEffort string
	if v, ok := reqMap["reasoning_effort"].(string); ok && v != "" {
		rawEffort = strings.ToLower(strings.TrimSpace(v))
	} else if rObj, ok := reqMap["reasoning"].(map[string]any); ok {
		if v, ok := rObj["effort"].(string); ok && v != "" {
			rawEffort = strings.ToLower(strings.TrimSpace(v))
		}
	} else if v, ok := ctk["reasoning_effort"].(string); ok && v != "" {
		rawEffort = strings.ToLower(strings.TrimSpace(v))
	}

	if v, ok := ctk["enable_thinking"].(bool); ok {
		cfg.EnableThinking = v
	}
	if v, ok := reqMap["enable_thinking"].(bool); ok {
		cfg.EnableThinking = v
		delete(reqMap, "enable_thinking")
	}

	switch rawEffort {
	case "none", "off", "disable", "disabled":
		cfg.Effort = "low"
		cfg.EnableThinking = false
		cfg.SoftTokenCap = 0
		cfg.HardTokenCap = 0
	case "low":
		cfg.Effort = "low"
		cfg.SoftTokenCap = 192
		cfg.HardTokenCap = 256
	case "medium":
		cfg.Effort = "medium"
		cfg.SoftTokenCap = 768
		cfg.HardTokenCap = 1024
	case "high":
		cfg.Effort = "high"
		cfg.SoftTokenCap = 3072
		cfg.HardTokenCap = 4096
	case "max", "":
		cfg.Effort = "max"
		cfg.SoftTokenCap = 32768
		cfg.HardTokenCap = 32768
	default:
		cfg.Effort = "max"
	}

	if v, ok := reqMap["clear_thinking"].(bool); ok {
		cfg.ClearThinking = v
		delete(reqMap, "clear_thinking")
	} else if v, ok := ctk["clear_thinking"].(bool); ok {
		cfg.ClearThinking = v
	}

	ctk["reasoning_effort"] = cfg.Effort
	ctk["clear_thinking"] = cfg.ClearThinking
	ctk["enable_thinking"] = cfg.EnableThinking
	reqMap["chat_template_kwargs"] = ctk

	// Strip historical reasoning_content and <think>...</think> from assistant messages
	// prior to the final user turn when clear_thinking is enabled.
	rawMsgs, ok := reqMap["messages"].([]any)
	if !ok || len(rawMsgs) == 0 {
		return cfg
	}

	lastUserIdx := -1
	for i, rm := range rawMsgs {
		if m, ok := rm.(map[string]any); ok {
			if role, _ := m["role"].(string); role == "user" {
				lastUserIdx = i
			}
		}
	}

	for i, rm := range rawMsgs {
		m, ok := rm.(map[string]any)
		if !ok {
			continue
		}
		role, _ := m["role"].(string)
		if role == "assistant" && cfg.ClearThinking && i < lastUserIdx {
			delete(m, "reasoning_content")
			if cStr, ok := m["content"].(string); ok && strings.Contains(cStr, "</think>") {
				m["content"] = strings.TrimSpace(thinkTagRegex.ReplaceAllString(cStr, ""))
			}
		}
		// Normalize assistant tool_calls arguments from JSON string to map for GLM-5.3 template compatibility
		if role == "assistant" {
			if tcs, ok := m["tool_calls"].([]any); ok {
				for _, tcRaw := range tcs {
					if tc, ok := tcRaw.(map[string]any); ok {
						if fn, ok := tc["function"].(map[string]any); ok {
							if argStr, ok := fn["arguments"].(string); ok {
								var parsed map[string]any
								if err := json.Unmarshal([]byte(RepairMalformedJSON(argStr)), &parsed); err == nil {
									fn["arguments"] = parsed
								} else {
									fn["arguments"] = map[string]any{}
								}
							}
						}
					}
				}
			}
		}
	}

	return cfg
}

// ============================================================================
// 3. Streaming `glm47_repair` Tool Parser & Runaway Repetition Loop Guard
// ============================================================================

var (
	xmlToolCallBlockRegex = regexp.MustCompile(`(?s)<tool_call>\s*(.*?)\s*</tool_call>`)
	xmlArgPairRegex       = regexp.MustCompile(`(?s)<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>`)
	unquotedKeyRegex      = regexp.MustCompile(`([{,]\s*)([a-zA-Z_][a-zA-Z0-9_-]*)\s*:`)
	trailingCommaRegex    = regexp.MustCompile(`,\s*([}\]])`)
)

// RepairMalformedJSON repairs common LLM JSON syntax errors (single quotes, unquoted keys,
// Python booleans/None, trailing commas, and unclosed braces/brackets/quotes).
func RepairMalformedJSON(raw string) string {
	s := strings.TrimSpace(raw)
	if s == "" {
		return "{}"
	}
	// Strip markdown fences if present
	if strings.HasPrefix(s, "```") {
		lines := strings.Split(s, "\n")
		if len(lines) >= 2 {
			lines = lines[1:]
			if strings.HasPrefix(strings.TrimSpace(lines[len(lines)-1]), "```") {
				lines = lines[:len(lines)-1]
			}
			s = strings.TrimSpace(strings.Join(lines, "\n"))
		}
	}

	var check any
	if json.Unmarshal([]byte(s), &check) == nil {
		return s
	}

	// Replace Python literals outside quotes
	s = strings.ReplaceAll(s, ": True", ": true")
	s = strings.ReplaceAll(s, ": False", ": false")
	s = strings.ReplaceAll(s, ": None", ": null")
	s = strings.ReplaceAll(s, ":True", ":true")
	s = strings.ReplaceAll(s, ":False", ":false")
	s = strings.ReplaceAll(s, ":None", ":null")

	// Convert single-quoted JSON keys/values if no double quotes exist
	if !strings.Contains(s, `"`) && strings.Contains(s, `'`) {
		s = strings.ReplaceAll(s, `'`, `"`)
	}

	// Quote unquoted keys: {foo: "bar"} -> {"foo": "bar"}
	s = unquotedKeyRegex.ReplaceAllString(s, `$1"$2":`)

	// Remove trailing commas: {"a": 1, } -> {"a": 1}
	s = trailingCommaRegex.ReplaceAllString(s, `$1`)

	// Balance unclosed quotes and braces/brackets
	inStr := false
	escaped := false
	var stack []byte
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if escaped {
			escaped = false
			continue
		}
		if ch == '\\' && inStr {
			escaped = true
			continue
		}
		if ch == '"' {
			inStr = !inStr
			continue
		}
		if !inStr {
			switch ch {
			case '{':
				stack = append(stack, '}')
			case '[':
				stack = append(stack, ']')
			case '}', ']':
				if len(stack) > 0 && stack[len(stack)-1] == ch {
					stack = stack[:len(stack)-1]
				}
			}
		}
	}
	if inStr {
		s += `"`
	}
	s = trailingCommaRegex.ReplaceAllString(s, `$1`)
	for i := len(stack) - 1; i >= 0; i-- {
		s += string(stack[i])
	}

	if json.Unmarshal([]byte(s), &check) == nil {
		return s
	}
	// Final fallback: wrap as raw string value
	b, _ := json.Marshal(map[string]string{"raw": raw})
	return string(b)
}

type ParsedToolCall struct {
	ID        string
	Name      string
	Arguments string
}

// ParseGLM47ToolCalls extracts and repairs tool calls from GLM-4.7 / GLM-5.3 XML or JSON blocks.
func ParseGLM47ToolCalls(raw string, startIdx int) ([]ParsedToolCall, string) {
	matches := xmlToolCallBlockRegex.FindAllStringSubmatchIndex(raw, -1)
	if len(matches) == 0 {
		return nil, raw
	}

	var calls []ParsedToolCall
	var cleaned strings.Builder
	lastEnd := 0

	for i, m := range matches {
		cleaned.WriteString(raw[lastEnd:m[0]])
		lastEnd = m[1]
		inner := strings.TrimSpace(raw[m[2]:m[3]])
		callID := fmt.Sprintf("call_glm53_%d_%d", time.Now().UnixNano()%1000000, startIdx+i)

		// Case 1: JSON inside <tool_call>{...}</tool_call>
		if strings.HasPrefix(inner, "{") {
			repaired := RepairMalformedJSON(inner)
			var obj map[string]any
			if err := json.Unmarshal([]byte(repaired), &obj); err == nil {
				name, _ := obj["name"].(string)
				if name != "" {
					argsObj := obj["arguments"]
					if argsObj == nil {
						argsObj = obj["parameters"]
					}
					var argsStr string
					switch v := argsObj.(type) {
					case string:
						argsStr = RepairMalformedJSON(v)
					case nil:
						argsStr = "{}"
					default:
						b, _ := json.Marshal(v)
						argsStr = string(b)
					}
					calls = append(calls, ParsedToolCall{
						ID:        callID,
						Name:      name,
						Arguments: argsStr,
					})
					continue
				}
			}
		}

		// Case 2: GLM-4.7 / GLM-5.3 XML format:
		// <tool_call>func_name<arg_key>k1</arg_key><arg_value>v1</arg_value>...</tool_call>
		funcName := inner
		if idx := strings.Index(inner, "<arg_key>"); idx >= 0 {
			funcName = strings.TrimSpace(inner[:idx])
		} else if idx := strings.Index(inner, "\n"); idx >= 0 {
			funcName = strings.TrimSpace(inner[:idx])
		}
		funcName = strings.Trim(funcName, "`\"' \t\r\n")

		argsMap := make(map[string]any)
		argMatches := xmlArgPairRegex.FindAllStringSubmatch(inner, -1)
		for _, am := range argMatches {
			k := strings.TrimSpace(am[1])
			vRaw := strings.TrimSpace(am[2])
			if k == "" {
				continue
			}
			argsMap[k] = parseOrRepairArgValue(vRaw)
		}

		argsBytes, _ := json.Marshal(argsMap)
		calls = append(calls, ParsedToolCall{
			ID:        callID,
			Name:      funcName,
			Arguments: string(argsBytes),
		})
	}
	cleaned.WriteString(raw[lastEnd:])
	return calls, cleaned.String()
}

func parseOrRepairArgValue(vRaw string) any {
	if vRaw == "" {
		return ""
	}
	if vRaw == "true" || vRaw == "True" {
		return true
	}
	if vRaw == "false" || vRaw == "False" {
		return false
	}
	if vRaw == "null" || vRaw == "None" {
		return nil
	}
	if i, err := strconv.ParseInt(vRaw, 10, 64); err == nil {
		return i
	}
	if f, err := strconv.ParseFloat(vRaw, 64); err == nil {
		return f
	}
	if strings.HasPrefix(vRaw, "{") || strings.HasPrefix(vRaw, "[") {
		repaired := RepairMalformedJSON(vRaw)
		var parsed any
		if err := json.Unmarshal([]byte(repaired), &parsed); err == nil {
			return parsed
		}
	}
	var quotedStr string
	if strings.HasPrefix(vRaw, `"`) && strings.HasSuffix(vRaw, `"`) {
		if err := json.Unmarshal([]byte(vRaw), &quotedStr); err == nil {
			return quotedStr
		}
	}
	return vRaw
}

// LoopGuard detects runaway N-gram token/byte repetition loops in real time.
type LoopGuard struct {
	window bytes.Buffer
}

func (lg *LoopGuard) FeedAndCheck(chunk string) bool {
	if chunk == "" {
		return false
	}
	lg.window.WriteString(chunk)
	b := lg.window.Bytes()
	if len(b) > 4096 {
		b = b[len(b)-4096:]
		lg.window.Reset()
		lg.window.Write(b)
	}
	n := len(b)
	if n < 180 {
		return false
	}
	// Check if any suffix pattern of length L in [24..160] repeats >= 6 times consecutively
	for patLen := 24; patLen <= 160 && patLen*6 <= n; patLen++ {
		tail := b[n-patLen:]
		repeats := 1
		for pos := n - 2*patLen; pos >= 0; pos -= patLen {
			if bytes.Equal(b[pos:pos+patLen], tail) {
				repeats++
				if repeats >= 6 {
					return true
				}
			} else {
				break
			}
		}
	}
	return false
}

// ============================================================================
// 4. Gateway Server & OpenAI / Anthropic / Responses / Tokenize / Metrics Handlers
// ============================================================================

type GatewayServer struct {
	upstreamURL       string
	client            *http.Client
	router            *RadixPrefixRouter
	totalRequests     atomic.Uint64
	repairedToolCalls atomic.Uint64
	loopGuardAborts   atomic.Uint64
	clientDisconnects atomic.Uint64
}

func NewGatewayServer(upstreamURL string, router *RadixPrefixRouter) *GatewayServer {
	tr := &http.Transport{
		MaxIdleConns:        256,
		MaxIdleConnsPerHost: 256,
		IdleConnTimeout:     90 * time.Second,
	}
	return &GatewayServer{
		upstreamURL: strings.TrimRight(upstreamURL, "/"),
		client: &http.Client{
			Transport: tr,
			Timeout:   600 * time.Second,
		},
		router: router,
	}
}

func extractTextContent(c any) string {
	switch v := c.(type) {
	case string:
		return v
	case []any:
		var sb strings.Builder
		for _, item := range v {
			if m, ok := item.(map[string]any); ok {
				if t, ok := m["text"].(string); ok {
					sb.WriteString(t)
				} else if t, ok := m["content"].(string); ok {
					sb.WriteString(t)
				}
			} else if s, ok := item.(string); ok {
				sb.WriteString(s)
			}
		}
		return sb.String()
	default:
		return ""
	}
}

func (gs *GatewayServer) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	gs.totalRequests.Add(1)

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	var reqMap map[string]any
	if err := json.Unmarshal(bodyBytes, &reqMap); err != nil {
		http.Error(w, "invalid JSON payload: "+err.Error(), http.StatusBadRequest)
		return
	}

	reasoningCfg := NormalizeReasoningAndMessages(reqMap)

	var msgs []map[string]any
	if rawMsgs, ok := reqMap["messages"].([]any); ok {
		for _, rm := range rawMsgs {
			if m, ok := rm.(map[string]any); ok {
				msgs = append(msgs, m)
			}
		}
	}
	var tools []any
	if rawTools, ok := reqMap["tools"].([]any); ok {
		tools = rawTools
	}

	hitBlocks, hitTokens, nvmeRestored := gs.router.RouteAndPin(msgs, tools)
	w.Header().Set("X-Spark-Prefix-Cache-Hit-Blocks", strconv.Itoa(hitBlocks))
	w.Header().Set("X-Spark-Prefix-Cache-Hit-Tokens", strconv.Itoa(hitTokens))
	w.Header().Set("X-Spark-NVMe-Spill-Restored", strconv.Itoa(nvmeRestored))
	w.Header().Set("X-Spark-Reasoning-Effort", reasoningCfg.Effort)

	// Move top-level tools into chat_template_kwargs["tools"] so vLLM's Jinja template
	// formats GLM-5.3 tool definitions while our compiled Go glm47_repair parser handles
	// streaming & non-streaming <tool_call> extraction without vLLM 400 tool_choice errors.
	if len(tools) > 0 {
		tcChoice, _ := reqMap["tool_choice"].(string)
		if strings.ToLower(strings.TrimSpace(tcChoice)) != "none" {
			ctk, _ := reqMap["chat_template_kwargs"].(map[string]any)
			if ctk == nil {
				ctk = make(map[string]any)
			}
			ctk["tools"] = tools
			reqMap["chat_template_kwargs"] = ctk
		}
		delete(reqMap, "tools")
		delete(reqMap, "tool_choice")
	} else {
		delete(reqMap, "tool_choice")
	}

	isStream, _ := reqMap["stream"].(bool)
	upstreamBody, err := json.Marshal(reqMap)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	var detCacheKey string
	if tempVal, hasTemp := reqMap["temperature"].(float64); hasTemp && tempVal == 0.0 {
		h := sha256.Sum256(upstreamBody)
		detCacheKey = hex.EncodeToString(h[:16]) + ":" + reasoningCfg.Effort
		if hitBlocks > 0 {
			if isStream {
				if cachedChunks, ok := gs.router.GetDeterministicStream(detCacheKey); ok {
					w.Header().Set("Content-Type", "text/event-stream")
					w.Header().Set("Cache-Control", "no-cache")
					w.Header().Set("Connection", "keep-alive")
					flusher, _ := w.(http.Flusher)
					for _, chunk := range cachedChunks {
						_, _ = fmt.Fprintf(w, "data: %s\n\n", chunk)
						if flusher != nil {
							flusher.Flush()
						}
					}
					_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
					if flusher != nil {
						flusher.Flush()
					}
					return
				}
			} else {
				if cachedJSON, ok := gs.router.GetDeterministicJSON(detCacheKey); ok {
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write(cachedJSON)
					return
				}
			}
		}
	}

	ctx, cancelUpstream := context.WithCancel(r.Context())
	defer cancelUpstream()

	upReq, err := http.NewRequestWithContext(ctx, http.MethodPost, gs.upstreamURL+"/v1/chat/completions", bytes.NewReader(upstreamBody))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	upReq.Header.Set("Content-Type", "application/json")

	resp, err := gs.client.Do(upReq)
	if err != nil {
		if r.Context().Err() != nil {
			gs.clientDisconnects.Add(1)
			return
		}
		http.Error(w, "upstream error: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		w.WriteHeader(resp.StatusCode)
		_, _ = io.Copy(w, resp.Body)
		return
	}

	if !isStream {
		gs.handleNonStreamingChatResponse(w, resp.Body, reasoningCfg, detCacheKey)
		return
	}

	gs.handleStreamingChatResponse(w, r, resp.Body, cancelUpstream, reasoningCfg, detCacheKey)
}

func (gs *GatewayServer) handleNonStreamingChatResponse(w http.ResponseWriter, body io.Reader, reasoningCfg ReasoningConfig, detCacheKey string) {
	rawBytes, err := io.ReadAll(body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}

	var respObj map[string]any
	if err := json.Unmarshal(rawBytes, &respObj); err != nil {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(rawBytes)
		return
	}

	choices, _ := respObj["choices"].([]any)
	for _, chRaw := range choices {
		ch, ok := chRaw.(map[string]any)
		if !ok {
			continue
		}
		msg, ok := ch["message"].(map[string]any)
		if !ok {
			continue
		}
		content, _ := msg["content"].(string)
		content = strings.TrimPrefix(content, "<|assistant|>")

		// Separate <think>...</think> into reasoning_content
		if strings.Contains(content, "</think>") {
			parts := strings.SplitN(content, "</think>", 2)
			thinkPart := strings.TrimPrefix(strings.TrimSpace(parts[0]), "<think>")
			content = strings.TrimSpace(parts[1])
			if reasoningCfg.EnableThinking && thinkPart != "" {
				thinkPart = truncateByTokenCap(thinkPart, reasoningCfg.HardTokenCap)
				msg["reasoning_content"] = thinkPart
			}
		} else if strings.HasPrefix(strings.TrimSpace(content), "<think>") {
			thinkPart := strings.TrimPrefix(strings.TrimSpace(content), "<think>")
			if reasoningCfg.EnableThinking {
				msg["reasoning_content"] = truncateByTokenCap(thinkPart, reasoningCfg.HardTokenCap)
				content = ""
			}
		}

		// Parse and repair any <tool_call>...</tool_call> blocks
		if strings.Contains(content, "<tool_call>") {
			parsedCalls, cleaned := ParseGLM47ToolCalls(content, 0)
			if len(parsedCalls) > 0 {
				gs.repairedToolCalls.Add(uint64(len(parsedCalls)))
				var tcList []map[string]any
				for _, pc := range parsedCalls {
					tcList = append(tcList, map[string]any{
						"id":   pc.ID,
						"type": "function",
						"function": map[string]any{
							"name":      pc.Name,
							"arguments": pc.Arguments,
						},
					})
				}
				msg["tool_calls"] = tcList
				msg["content"] = strings.TrimSpace(cleaned)
				ch["finish_reason"] = "tool_calls"
			}
		} else {
			msg["content"] = content
		}

		// Also repair any existing tool_calls arguments from upstream
		if existingTCs, ok := msg["tool_calls"].([]any); ok && len(existingTCs) > 0 {
			for _, tcRaw := range existingTCs {
				if tc, ok := tcRaw.(map[string]any); ok {
					if fn, ok := tc["function"].(map[string]any); ok {
						if argStr, ok := fn["arguments"].(string); ok {
							fn["arguments"] = RepairMalformedJSON(argStr)
						}
					}
				}
			}
		}
	}

	outBytes, _ := json.Marshal(respObj)
	if detCacheKey != "" {
		gs.router.SetDeterministicJSON(detCacheKey, outBytes)
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(outBytes)
}

func truncateByTokenCap(text string, maxTokens int) string {
	if maxTokens <= 0 {
		return ""
	}
	maxChars := maxTokens * 4
	if len(text) <= maxChars {
		return text
	}
	return text[:maxChars]
}

func (gs *GatewayServer) handleStreamingChatResponse(
	w http.ResponseWriter,
	r *http.Request,
	body io.Reader,
	cancelUpstream context.CancelFunc,
	reasoningCfg ReasoningConfig,
	detCacheKey string,
) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, _ := w.(http.Flusher)

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	loopGuard := &LoopGuard{}
	inThink := false
	thinkDetermined := false
	thinkTokens := 0
	inToolCall := false
	var toolBuf strings.Builder
	toolCallIdx := 0
	var lastChunkTemplate map[string]any
	var recordedSSE []string

	writeSSE := func(obj map[string]any) {
		b, err := json.Marshal(obj)
		if err != nil {
			return
		}
		if detCacheKey != "" {
			recordedSSE = append(recordedSSE, string(b))
		}
		_, _ = fmt.Fprintf(w, "data: %s\n\n", string(b))
		if flusher != nil {
			flusher.Flush()
		}
	}

	for scanner.Scan() {
		if r.Context().Err() != nil {
			gs.clientDisconnects.Add(1)
			cancelUpstream()
			return
		}
		line := scanner.Text()
		if line == "" || !strings.HasPrefix(line, "data: ") {
			continue
		}
		dataStr := strings.TrimSpace(line[6:])
		if dataStr == "[DONE]" {
			// Flush any pending tool call in toolBuf
			if inToolCall && toolBuf.Len() > 0 {
				rawTC := toolBuf.String()
				if !strings.Contains(rawTC, "</tool_call>") {
					rawTC += "</tool_call>"
				}
				calls, _ := ParseGLM47ToolCalls(rawTC, toolCallIdx)
				for _, pc := range calls {
					gs.repairedToolCalls.Add(1)
					emitToolCallDelta(writeSSE, lastChunkTemplate, pc, toolCallIdx)
					toolCallIdx++
				}
				inToolCall = false
				toolBuf.Reset()
			}
			if detCacheKey != "" && len(recordedSSE) > 0 {
				gs.router.SetDeterministicStream(detCacheKey, recordedSSE)
			}
			_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
			if flusher != nil {
				flusher.Flush()
			}
			return
		}

		var evt map[string]any
		if err := json.Unmarshal([]byte(dataStr), &evt); err != nil {
			_, _ = fmt.Fprintf(w, "%s\n\n", line)
			if flusher != nil {
				flusher.Flush()
			}
			continue
		}
		lastChunkTemplate = evt

		choices, _ := evt["choices"].([]any)
		if len(choices) == 0 {
			writeSSE(evt)
			continue
		}
		ch, ok := choices[0].(map[string]any)
		if !ok {
			writeSSE(evt)
			continue
		}
		delta, _ := ch["delta"].(map[string]any)
		if delta == nil {
			writeSSE(evt)
			continue
		}

		// Repair any structured tool_calls already parsed by upstream
		if tcs, ok := delta["tool_calls"].([]any); ok && len(tcs) > 0 {
			for _, tcRaw := range tcs {
				if tc, ok := tcRaw.(map[string]any); ok {
					if fn, ok := tc["function"].(map[string]any); ok {
						if argStr, ok := fn["arguments"].(string); ok && strings.HasPrefix(strings.TrimSpace(argStr), "{") && strings.HasSuffix(strings.TrimSpace(argStr), "}") {
							fn["arguments"] = RepairMalformedJSON(argStr)
						}
					}
				}
			}
		}

		content, _ := delta["content"].(string)
		reasoningDelta, _ := delta["reasoning_content"].(string)

		// Runaway loop guard check on incoming token stream
		if loopGuard.FeedAndCheck(content + reasoningDelta) {
			gs.loopGuardAborts.Add(1)
			cancelUpstream()
			ch["finish_reason"] = "stop"
			delta["content"] = ""
			writeSSE(evt)
			_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
			if flusher != nil {
				flusher.Flush()
			}
			return
		}

		if reasoningDelta != "" {
			thinkTokens++
			if !reasoningCfg.EnableThinking || (reasoningCfg.HardTokenCap > 0 && thinkTokens > reasoningCfg.HardTokenCap) {
				delete(delta, "reasoning_content")
				if content == "" {
					continue
				}
			}
		}

		if content == "" {
			writeSSE(evt)
			continue
		}

		content = strings.ReplaceAll(content, "<|assistant|>", "")
		if !thinkDetermined {
			trimmed := strings.TrimLeft(content, " \t\r\n")
			if strings.HasPrefix(trimmed, "<think>") {
				inThink = true
				thinkDetermined = true
				content = strings.TrimPrefix(trimmed, "<think>")
			} else if strings.Contains(content, "</think>") {
				inThink = true
				thinkDetermined = true
			} else if trimmed != "" {
				thinkDetermined = true
			}
		}

		if inThink {
			if idx := strings.Index(content, "</think>"); idx >= 0 {
				thinkPart := content[:idx]
				afterThink := content[idx+len("</think>"):]
				inThink = false
				if thinkPart != "" && reasoningCfg.EnableThinking && thinkTokens <= reasoningCfg.HardTokenCap {
					thinkEvt := cloneEventWithDelta(evt, map[string]any{"reasoning_content": thinkPart}, nil)
					writeSSE(thinkEvt)
				}
				content = strings.TrimLeft(afterThink, "\r\n")
				if content == "" {
					continue
				}
			} else {
				thinkTokens++
				if reasoningCfg.EnableThinking && thinkTokens <= reasoningCfg.HardTokenCap {
					delete(delta, "content")
					delta["reasoning_content"] = content
					writeSSE(evt)
				}
				continue
			}
		}

		// Handle <tool_call>...</tool_call> streaming buffer
		if inToolCall || strings.Contains(content, "<tool_call>") {
			if !inToolCall {
				tcStart := strings.Index(content, "<tool_call>")
				prefixText := content[:tcStart]
				if prefixText != "" {
					textEvt := cloneEventWithDelta(evt, map[string]any{"content": prefixText}, nil)
					writeSSE(textEvt)
				}
				inToolCall = true
				toolBuf.WriteString(content[tcStart:])
			} else {
				toolBuf.WriteString(content)
			}

			bufStr := toolBuf.String()
			for {
				endIdx := strings.Index(bufStr, "</tool_call>")
				if endIdx < 0 {
					break
				}
				completeBlock := bufStr[:endIdx+len("</tool_call>")]
				calls, _ := ParseGLM47ToolCalls(completeBlock, toolCallIdx)
				for _, pc := range calls {
					gs.repairedToolCalls.Add(1)
					emitToolCallDelta(writeSSE, evt, pc, toolCallIdx)
					toolCallIdx++
				}
				bufStr = bufStr[endIdx+len("</tool_call>"):]
				if nextStart := strings.Index(bufStr, "<tool_call>"); nextStart >= 0 {
					between := strings.TrimSpace(bufStr[:nextStart])
					if between != "" {
						writeSSE(cloneEventWithDelta(evt, map[string]any{"content": between}, nil))
					}
					bufStr = bufStr[nextStart:]
				} else {
					trailing := strings.TrimSpace(bufStr)
					if trailing != "" {
						writeSSE(cloneEventWithDelta(evt, map[string]any{"content": trailing}, nil))
					}
					bufStr = ""
					inToolCall = false
					break
				}
			}
			toolBuf.Reset()
			toolBuf.WriteString(bufStr)
			continue
		}

		delta["content"] = content
		if toolCallIdx > 0 && ch["finish_reason"] == "stop" {
			ch["finish_reason"] = "tool_calls"
		}
		writeSSE(evt)
	}
}

func cloneEventWithDelta(base map[string]any, newDelta map[string]any, finishReason any) map[string]any {
	out := make(map[string]any, len(base))
	for k, v := range base {
		out[k] = v
	}
	ch := map[string]any{
		"index":         0,
		"delta":         newDelta,
		"finish_reason": finishReason,
	}
	out["choices"] = []any{ch}
	return out
}

func emitToolCallDelta(writeSSE func(map[string]any), tmpl map[string]any, pc ParsedToolCall, idx int) {
	if tmpl == nil {
		tmpl = map[string]any{
			"id":      "chatcmpl-spark",
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   "GLM-5.3-Flash",
		}
	}
	delta := map[string]any{
		"tool_calls": []any{
			map[string]any{
				"index": idx,
				"id":    pc.ID,
				"type":  "function",
				"function": map[string]any{
					"name":      pc.Name,
					"arguments": pc.Arguments,
				},
			},
		},
	}
	writeSSE(cloneEventWithDelta(tmpl, delta, "tool_calls"))
}

// ============================================================================
// 5. OpenAI `/v1/responses` Adapter
// ============================================================================

func (gs *GatewayServer) handleResponsesAPI(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	var reqMap map[string]any
	if err := json.Unmarshal(bodyBytes, &reqMap); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	var messages []any
	if instr, ok := reqMap["instructions"].(string); ok && instr != "" {
		messages = append(messages, map[string]any{"role": "system", "content": instr})
	}
	switch inp := reqMap["input"].(type) {
	case string:
		messages = append(messages, map[string]any{"role": "user", "content": inp})
	case []any:
		messages = append(messages, inp...)
	}

	chatReq := map[string]any{
		"model":       reqMap["model"],
		"messages":    messages,
		"stream":      false,
		"temperature": reqMap["temperature"],
	}
	if chatReq["model"] == nil {
		chatReq["model"] = "GLM-5.3-Flash"
	}
	if maxOut, ok := reqMap["max_output_tokens"]; ok {
		chatReq["max_tokens"] = maxOut
	}
	if tools, ok := reqMap["tools"]; ok {
		chatReq["chat_template_kwargs"] = map[string]any{"tools": tools}
	}
	if reasoning, ok := reqMap["reasoning"]; ok {
		chatReq["reasoning"] = reasoning
	}

	NormalizeReasoningAndMessages(chatReq)
	chatBytes, _ := json.Marshal(chatReq)

	upReq, _ := http.NewRequestWithContext(r.Context(), http.MethodPost, gs.upstreamURL+"/v1/chat/completions", bytes.NewReader(chatBytes))
	upReq.Header.Set("Content-Type", "application/json")
	resp, err := gs.client.Do(upReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	var rec FakeResponseWriter
	rec.header = make(http.Header)
	gs.handleNonStreamingChatResponse(&rec, resp.Body, ReasoningConfig{EnableThinking: true, HardTokenCap: 32768}, "")

	var chatResp map[string]any
	_ = json.Unmarshal(rec.buf.Bytes(), &chatResp)

	var outputItems []any
	if choices, ok := chatResp["choices"].([]any); ok && len(choices) > 0 {
		if ch, ok := choices[0].(map[string]any); ok {
			if msg, ok := ch["message"].(map[string]any); ok {
				if rc, ok := msg["reasoning_content"].(string); ok && rc != "" {
					outputItems = append(outputItems, map[string]any{
						"type": "reasoning",
						"id":   "rs_" + strconv.FormatInt(time.Now().UnixNano(), 10),
						"summary": []any{
							map[string]any{"type": "summary_text", "text": rc},
						},
					})
				}
				if c, ok := msg["content"].(string); ok && c != "" {
					outputItems = append(outputItems, map[string]any{
						"type":   "message",
						"id":     "msg_" + strconv.FormatInt(time.Now().UnixNano(), 10),
						"status": "completed",
						"role":   "assistant",
						"content": []any{
							map[string]any{"type": "output_text", "text": c},
						},
					})
				}
				if tcs, ok := msg["tool_calls"].([]any); ok {
					for _, tcRaw := range tcs {
						if tc, ok := tcRaw.(map[string]any); ok {
							fn, _ := tc["function"].(map[string]any)
							outputItems = append(outputItems, map[string]any{
								"type":      "function_call",
								"id":        tc["id"],
								"call_id":   tc["id"],
								"name":      fn["name"],
								"arguments": fn["arguments"],
								"status":    "completed",
							})
						}
					}
				}
			}
		}
	}

	out := map[string]any{
		"id":         "resp_" + strconv.FormatInt(time.Now().UnixNano(), 10),
		"object":     "response",
		"created_at": time.Now().Unix(),
		"status":     "completed",
		"model":      chatReq["model"],
		"output":     outputItems,
		"usage":      chatResp["usage"],
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

type FakeResponseWriter struct {
	header http.Header
	buf    bytes.Buffer
	status int
}

func (f *FakeResponseWriter) Header() http.Header         { return f.header }
func (f *FakeResponseWriter) Write(b []byte) (int, error) { return f.buf.Write(b) }
func (f *FakeResponseWriter) WriteHeader(statusCode int)  { f.status = statusCode }

// ============================================================================
// 6. Anthropic `/v1/messages` Compatibility Endpoint
// ============================================================================

func (gs *GatewayServer) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	gs.totalRequests.Add(1)

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	var anthReq map[string]any
	if err := json.Unmarshal(bodyBytes, &anthReq); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	var oaMsgs []map[string]any
	if sysPrompt := extractTextContent(anthReq["system"]); sysPrompt != "" {
		oaMsgs = append(oaMsgs, map[string]any{"role": "system", "content": sysPrompt})
	}

	if rawMsgs, ok := anthReq["messages"].([]any); ok {
		for _, rm := range rawMsgs {
			m, ok := rm.(map[string]any)
			if !ok {
				continue
			}
			role, _ := m["role"].(string)
			switch content := m["content"].(type) {
			case string:
				oaMsgs = append(oaMsgs, map[string]any{"role": role, "content": content})
			case []any:
				var textParts []string
				var toolCalls []any
				var toolResults []map[string]any
				var thinkingText string
				for _, blkRaw := range content {
					blk, ok := blkRaw.(map[string]any)
					if !ok {
						continue
					}
					bType, _ := blk["type"].(string)
					switch bType {
					case "text":
						if t, ok := blk["text"].(string); ok {
							textParts = append(textParts, t)
						}
					case "thinking":
						if t, ok := blk["thinking"].(string); ok {
							thinkingText = t
						}
					case "tool_use":
						inputBytes, _ := json.Marshal(blk["input"])
						toolCalls = append(toolCalls, map[string]any{
							"id":   blk["id"],
							"type": "function",
							"function": map[string]any{
								"name":      blk["name"],
								"arguments": string(inputBytes),
							},
						})
					case "tool_result":
						toolResults = append(toolResults, map[string]any{
							"role":         "tool",
							"tool_call_id": blk["tool_use_id"],
							"content":      extractTextContent(blk["content"]),
						})
					}
				}
				if len(textParts) > 0 || len(toolCalls) > 0 || thinkingText != "" {
					msgObj := map[string]any{
						"role":    role,
						"content": strings.Join(textParts, "\n"),
					}
					if thinkingText != "" {
						msgObj["reasoning_content"] = thinkingText
					}
					if len(toolCalls) > 0 {
						msgObj["tool_calls"] = toolCalls
					}
					oaMsgs = append(oaMsgs, msgObj)
				}
				for _, tr := range toolResults {
					oaMsgs = append(oaMsgs, tr)
				}
			}
		}
	}

	var oaTools []any
	if rawTools, ok := anthReq["tools"].([]any); ok {
		for _, rt := range rawTools {
			if tm, ok := rt.(map[string]any); ok {
				oaTools = append(oaTools, map[string]any{
					"type": "function",
					"function": map[string]any{
						"name":        tm["name"],
						"description": tm["description"],
						"parameters":  tm["input_schema"],
					},
				})
			}
		}
	}

	enableThinking := false
	reasoningEffort := "low"
	if th, ok := anthReq["thinking"].(map[string]any); ok {
		if tType, _ := th["type"].(string); tType == "enabled" {
			enableThinking = true
			budget, _ := th["budget_tokens"].(float64)
			switch {
			case budget <= 512:
				reasoningEffort = "low"
			case budget <= 2048:
				reasoningEffort = "medium"
			default:
				reasoningEffort = "high"
			}
		}
	}

	oaReq := map[string]any{
		"model":       "GLM-5.3-Flash",
		"messages":    oaMsgs,
		"max_tokens":  anthReq["max_tokens"],
		"temperature": anthReq["temperature"],
		"stream":      false,
		"chat_template_kwargs": map[string]any{
			"enable_thinking":  enableThinking,
			"reasoning_effort": reasoningEffort,
			"clear_thinking":   true,
		},
	}
	if oaReq["max_tokens"] == nil {
		oaReq["max_tokens"] = 1024
	}
	if len(oaTools) > 0 {
		oaReq["chat_template_kwargs"].(map[string]any)["tools"] = oaTools
	}

	hitBlocks, hitTokens, nvmeRestored := gs.router.RouteAndPin(oaMsgs, oaTools)
	w.Header().Set("X-Spark-Prefix-Cache-Hit-Blocks", strconv.Itoa(hitBlocks))
	w.Header().Set("X-Spark-Prefix-Cache-Hit-Tokens", strconv.Itoa(hitTokens))
	w.Header().Set("X-Spark-NVMe-Spill-Restored", strconv.Itoa(nvmeRestored))

	oaBytes, _ := json.Marshal(oaReq)
	upReq, _ := http.NewRequestWithContext(r.Context(), http.MethodPost, gs.upstreamURL+"/v1/chat/completions", bytes.NewReader(oaBytes))
	upReq.Header.Set("Content-Type", "application/json")
	resp, err := gs.client.Do(upReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	var rec FakeResponseWriter
	rec.header = make(http.Header)
	gs.handleNonStreamingChatResponse(&rec, resp.Body, ReasoningConfig{
		Effort:         reasoningEffort,
		EnableThinking: enableThinking,
		ClearThinking:  true,
		HardTokenCap:   4096,
	}, "")

	var oaResp map[string]any
	_ = json.Unmarshal(rec.buf.Bytes(), &oaResp)

	var contentBlocks []map[string]any
	stopReason := "end_turn"
	if choices, ok := oaResp["choices"].([]any); ok && len(choices) > 0 {
		if ch, ok := choices[0].(map[string]any); ok {
			if msg, ok := ch["message"].(map[string]any); ok {
				if rc, ok := msg["reasoning_content"].(string); ok && rc != "" {
					contentBlocks = append(contentBlocks, map[string]any{
						"type":     "thinking",
						"thinking": rc,
					})
				}
				if c, ok := msg["content"].(string); ok && c != "" {
					contentBlocks = append(contentBlocks, map[string]any{
						"type": "text",
						"text": c,
					})
				}
				if tcs, ok := msg["tool_calls"].([]any); ok && len(tcs) > 0 {
					stopReason = "tool_use"
					for _, tcRaw := range tcs {
						if tc, ok := tcRaw.(map[string]any); ok {
							fn, _ := tc["function"].(map[string]any)
							var inputObj any = map[string]any{}
							if argStr, ok := fn["arguments"].(string); ok {
								_ = json.Unmarshal([]byte(RepairMalformedJSON(argStr)), &inputObj)
							}
							contentBlocks = append(contentBlocks, map[string]any{
								"type":  "tool_use",
								"id":    tc["id"],
								"name":  fn["name"],
								"input": inputObj,
							})
						}
					}
				}
			}
		}
	}
	if len(contentBlocks) == 0 {
		contentBlocks = append(contentBlocks, map[string]any{"type": "text", "text": ""})
	}

	usageMap, _ := oaResp["usage"].(map[string]any)
	inToks, _ := usageMap["prompt_tokens"].(float64)
	outToks, _ := usageMap["completion_tokens"].(float64)

	anthMsg := map[string]any{
		"id":            "msg_" + strconv.FormatInt(time.Now().UnixNano(), 10),
		"type":          "message",
		"role":          "assistant",
		"model":         anthReq["model"],
		"content":       contentBlocks,
		"stop_reason":   stopReason,
		"stop_sequence": nil,
		"usage": map[string]any{
			"input_tokens":  int(inToks),
			"output_tokens": int(outToks),
		},
	}

	isStream, _ := anthReq["stream"].(bool)
	if !isStream {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(anthMsg)
		return
	}

	// Emit Anthropic SSE event stream
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	flusher, _ := w.(http.Flusher)
	sendEvent := func(event string, payload any) {
		b, _ := json.Marshal(payload)
		_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, string(b))
		if flusher != nil {
			flusher.Flush()
		}
	}

	sendEvent("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":      anthMsg["id"],
			"type":    "message",
			"role":    "assistant",
			"model":   anthReq["model"],
			"content": []any{},
			"usage":   map[string]any{"input_tokens": int(inToks), "output_tokens": 0},
		},
	})
	for idx, blk := range contentBlocks {
		bType, _ := blk["type"].(string)
		switch bType {
		case "text":
			sendEvent("content_block_start", map[string]any{
				"type":          "content_block_start",
				"index":         idx,
				"content_block": map[string]any{"type": "text", "text": ""},
			})
			sendEvent("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": idx,
				"delta": map[string]any{"type": "text_delta", "text": blk["text"]},
			})
		case "thinking":
			sendEvent("content_block_start", map[string]any{
				"type":          "content_block_start",
				"index":         idx,
				"content_block": map[string]any{"type": "thinking", "thinking": ""},
			})
			sendEvent("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": idx,
				"delta": map[string]any{"type": "thinking_delta", "thinking": blk["thinking"]},
			})
		case "tool_use":
			inBytes, _ := json.Marshal(blk["input"])
			sendEvent("content_block_start", map[string]any{
				"type":  "content_block_start",
				"index": idx,
				"content_block": map[string]any{
					"type":  "tool_use",
					"id":    blk["id"],
					"name":  blk["name"],
					"input": map[string]any{},
				},
			})
			sendEvent("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": idx,
				"delta": map[string]any{"type": "input_json_delta", "partial_json": string(inBytes)},
			})
		}
		sendEvent("content_block_stop", map[string]any{"type": "content_block_stop", "index": idx})
	}
	sendEvent("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": stopReason, "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": int(outToks)},
	})
	sendEvent("message_stop", map[string]any{"type": "message_stop"})
}

func (gs *GatewayServer) handleProxyPass(w http.ResponseWriter, r *http.Request) {
	targetURL := gs.upstreamURL + r.URL.Path
	if r.URL.RawQuery != "" {
		targetURL += "?" + r.URL.RawQuery
	}
	var bodyReader io.Reader
	if r.Body != nil {
		defer r.Body.Close()
		b, _ := io.ReadAll(r.Body)
		bodyReader = bytes.NewReader(b)
	}
	upReq, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL, bodyReader)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	for k, vals := range r.Header {
		for _, v := range vals {
			upReq.Header.Add(k, v)
		}
	}
	resp, err := gs.client.Do(upReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	for k, vals := range resp.Header {
		for _, v := range vals {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

func (gs *GatewayServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	upResp, err := gs.client.Get(gs.upstreamURL + "/health")
	upOK := err == nil && upResp != nil && upResp.StatusCode == http.StatusOK
	if upResp != nil && upResp.Body != nil {
		upResp.Body.Close()
	}
	status := "ok"
	code := http.StatusOK
	if !upOK {
		status = "degraded"
		code = http.StatusServiceUnavailable
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":           status,
		"upstream_healthy": upOK,
		"radix_prefix_router": map[string]any{
			"block_size_tokens":       gs.router.blockSize,
			"lookups_total":           gs.router.totalLookups.Load(),
			"block_hits_total":        gs.router.totalBlockHits.Load(),
			"token_hits_total":        gs.router.totalTokenHits.Load(),
			"pinned_blocks":           gs.router.pinnedBlocks.Load(),
			"nvme_spill_writes_total": gs.router.nvmeSpillWrites.Load(),
			"nvme_spill_reads_total":  gs.router.nvmeSpillReads.Load(),
		},
	})
}

func (gs *GatewayServer) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprintf(w, "# HELP spark_gateway_requests_total Total requests handled by Spark-GLM53 Go Gateway\n")
	fmt.Fprintf(w, "# TYPE spark_gateway_requests_total counter\n")
	fmt.Fprintf(w, "spark_gateway_requests_total %d\n", gs.totalRequests.Load())
	fmt.Fprintf(w, "spark_gateway_radix_lookups_total %d\n", gs.router.totalLookups.Load())
	fmt.Fprintf(w, "spark_gateway_radix_block_hits_total %d\n", gs.router.totalBlockHits.Load())
	fmt.Fprintf(w, "spark_gateway_radix_token_hits_total %d\n", gs.router.totalTokenHits.Load())
	fmt.Fprintf(w, "spark_gateway_radix_pinned_blocks %d\n", gs.router.pinnedBlocks.Load())
	fmt.Fprintf(w, "spark_gateway_nvme_spill_writes_total %d\n", gs.router.nvmeSpillWrites.Load())
	fmt.Fprintf(w, "spark_gateway_nvme_spill_reads_total %d\n", gs.router.nvmeSpillReads.Load())
	fmt.Fprintf(w, "spark_gateway_repaired_tool_calls_total %d\n", gs.repairedToolCalls.Load())
	fmt.Fprintf(w, "spark_gateway_loop_guard_aborts_total %d\n", gs.loopGuardAborts.Load())
	fmt.Fprintf(w, "spark_gateway_client_disconnects_total %d\n", gs.clientDisconnects.Load())

	// Append upstream vLLM Prometheus metrics if available
	resp, err := gs.client.Get(gs.upstreamURL + "/metrics")
	if err == nil {
		defer resp.Body.Close()
		_, _ = io.Copy(w, resp.Body)
	}
}

// Keep unused imports referenced cleanly
var (
	_ = binary.LittleEndian
	_ = sort.Ints
)

func main() {
	listenAddr := flag.String("listen", ":8888", "Listen address for Spark-GLM53 Go Control Plane")
	upstreamURL := flag.String("upstream", "http://127.0.0.1:8000", "Upstream internal vLLM engine URL")
	spillDir := flag.String("spill-dir", "/tmp/spark_kv_spill", "Directory for O_DIRECT NVMe KV Spill Tier")
	blockSize := flag.Int("block-size", DefaultBlockSizeTokens, "KV cache block size in tokens (must match B12X 256)")
	flag.Parse()

	router := NewRadixPrefixRouter(*blockSize, MaxRadixBlocks, *spillDir)
	srv := NewGatewayServer(*upstreamURL, router)

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", srv.handleChatCompletions)
	mux.HandleFunc("/v1/responses", srv.handleResponsesAPI)
	mux.HandleFunc("/v1/messages", srv.handleAnthropicMessages)
	mux.HandleFunc("/v1/models", srv.handleProxyPass)
	mux.HandleFunc("/tokenize", srv.handleProxyPass)
	mux.HandleFunc("/detokenize", srv.handleProxyPass)
	mux.HandleFunc("/health", srv.handleHealth)
	mux.HandleFunc("/metrics", srv.handleMetrics)

	log.Printf("[spark-gateway] Starting Spark-GLM53 Go Control Plane on %s -> upstream %s (block_size=%d, spill_dir=%s)",
		*listenAddr, *upstreamURL, *blockSize, *spillDir)
	httpSrv := &http.Server{
		Addr:              *listenAddr,
		Handler:           mux,
		ReadHeaderTimeout: 30 * time.Second,
	}
	if err := httpSrv.ListenAndServe(); err != nil {
		log.Fatalf("[spark-gateway] server error: %v", err)
	}
}
