# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI entrypoint for the support-ops environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import SupportOpsAction, SupportOpsObservation
    from .support_ops_env_environment import SupportOpsEnvironment
except ModuleNotFoundError:
    from models import SupportOpsAction, SupportOpsObservation
    from server.support_ops_env_environment import SupportOpsEnvironment


app = create_app(
    SupportOpsEnvironment,
    SupportOpsAction,
    SupportOpsObservation,
    env_name="support_ops_env",
    max_concurrent_envs=4,
)


def main():
    """Run the environment server on port 8000."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
