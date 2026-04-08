# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Local smoke test for Support Ops Env."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_command(args: list[str], *, cwd: Path | None = None) -> None:
    """Run a command and stream output."""

    subprocess.run(args, cwd=cwd or ROOT, check=True)


def main() -> None:
    """Run a lightweight verification pass for the environment."""

    run_command([sys.executable, "-m", "pytest"])
    run_command(["uvx", "--from", "openenv-core", "openenv", "validate", "."])

    server = subprocess.Popen(
        [sys.executable, "-m", "support_ops_env.server.app"],
        cwd=ROOT,
    )
    try:
        time.sleep(6)
        run_command(
            [
                "uvx",
                "--from",
                "openenv-core",
                "openenv",
                "validate",
                "http://127.0.0.1:8000",
            ]
        )
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
