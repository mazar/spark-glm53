"""Auto-configure Spark-GLM53 environment variables and b12x compiler hooks in all spawned child processes."""

import os

if os.environ.get("SPARK_GLM53_ACTIVE", "1") == "1":
    try:
        from worker.b12x_dflash_runner import configure_spark_env, install_b12x_compiler_worker_patches

        configure_spark_env()
        install_b12x_compiler_worker_patches()
    except Exception:
        pass
