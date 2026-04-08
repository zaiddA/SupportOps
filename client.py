# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for talking to a running Support Ops environment."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import SupportOpsAction, SupportOpsObservation, SupportOpsState
except ImportError:
    from models import SupportOpsAction, SupportOpsObservation, SupportOpsState


class SupportOpsEnv(EnvClient[SupportOpsAction, SupportOpsObservation, SupportOpsState]):
    """Typed client for the OpenEnv HTTP/WebSocket server."""

    def _step_payload(self, action: SupportOpsAction) -> dict[str, Any]:
        """Serialize one action for the server."""

        return action.model_dump(mode="json", exclude_none=False)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[SupportOpsObservation]:
        """Turn a server step response back into typed objects."""

        obs_data = dict(payload.get("observation", {}))
        observation = SupportOpsObservation.model_validate(
            {
                **obs_data,
                "done": payload.get("done", False),
                "reward": payload.get("reward"),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> SupportOpsState:
        """Parse a state snapshot from the server."""

        return SupportOpsState.model_validate(payload)
