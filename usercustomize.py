"""Auto-configure Spark-GLM53 environment variables and b12x compiler hooks in spawned child processes."""

import os

if os.environ.get("SPARK_GLM53_ACTIVE", "1") == "1":
    try:
        with open("/proc/self/cmdline", "rb") as _f:
            _cmdline = _f.read().decode("utf-8", errors="ignore")
    except Exception:
        _cmdline = ""

    if "resource_tracker" not in _cmdline:
        try:
            from worker.b12x_dflash_runner import (
                configure_spark_env,
                install_b12x_compiler_worker_patches,
            )

            configure_spark_env()
            # Only load heavy b12x/cutlass compiler patches inside offline b12x CompilePool workers
            if (
                os.environ.get("CUDA_VISIBLE_DEVICES") == ""
                and "spawn_main" in _cmdline
            ):
                install_b12x_compiler_worker_patches()
        except Exception:
            pass


