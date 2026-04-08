# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small tuple-style wrapper around the environment implementation."""

from __future__ import annotations

try:
    from .models import SupportOpsAction, SupportOpsObservation, SupportOpsState
except ImportError:
    from models import SupportOpsAction, SupportOpsObservation, SupportOpsState

try:
    from .server.support_ops_env_environment import SupportOpsEnvironment
except ImportError:
    from server.support_ops_env_environment import SupportOpsEnvironment


class SupportOpsTrainerEnv:
    """Convenience wrapper for local training and scripted rollouts."""

    def __init__(self, task_id: str | None = None):
        self._env = SupportOpsEnvironment(default_task_id=task_id)
        self._task_id = task_id

    def reset(self, task_id: str | None = None, seed: int | None = None) -> SupportOpsObservation:
        """Reset the environment and return the first observation."""

        selected_task = task_id or self._task_id
        return self._env.reset(seed=seed, task_id=selected_task)

    def step(
        self,
        action: SupportOpsAction,
    ) -> tuple[SupportOpsObservation, float, bool, dict]:
        """Run one action and return a tuple-style transition."""

        observation = self._env.step(action)
        reward = float(observation.reward or 0.0)
        done = bool(observation.done)
        info = dict(observation.metadata)
        return observation, reward, done, info

    def state(self) -> SupportOpsState:
        """Return the current typed state snapshot."""

        return self._env.state
