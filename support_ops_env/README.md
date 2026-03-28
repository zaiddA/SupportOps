---
title: Support Ops Env
colorFrom: blue
colorTo: cyan
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - support
  - rl
  - agents
---

# Support Ops Env

Support Ops Env is an OpenEnv environment for evaluating agents on support operations work inside a SaaS company. The agent works a realistic inbox: triaging tickets, routing them to the right team, applying policy, replying to customers, documenting internal notes, and handling duplicate or incident-related tickets without touching unrelated work.

## Motivation

A lot of agent benchmarks focus on games, coding puzzles, or narrow one-shot tasks. Support operations is a better fit for real-world evaluation because it mixes stateful decision-making, policy use, communication, prioritization, and safe record updates. Progress is measurable, mistakes are costly, and partial success still matters.

This environment is built around three ideas:

- realistic workflows instead of toy interactions
- deterministic graders with partial credit
- reward shaping that encourages progress and discourages careless actions

## Environment API

Support Ops Env implements the standard OpenEnv interface and also includes a local tuple-style adapter for training loops.

### OpenEnv interface

- `reset(task_id=..., seed=...) -> SupportOpsObservation`
- `step(SupportOpsAction) -> SupportOpsObservation`
- `state -> SupportOpsState`

### Local training adapter

`SupportOpsTrainerEnv` exposes:

- `reset(task_id=..., seed=...) -> SupportOpsObservation`
- `step(action) -> (observation, reward, done, info)`
- `state() -> SupportOpsState`

## Action space

The main action model is `SupportOpsAction`. Supported action types:

- `update_ticket`
- `reply_to_customer`
- `add_internal_note`
- `merge_ticket`
- `finish`

Key fields:

- `ticket_id`
- `status`
- `priority`
- `assigned_team`
- `tags_to_add`
- `tags_to_remove`
- `resolution_code`
- `response_text`
- `note_text`
- `target_ticket_id`

## Observation space

`SupportOpsObservation` includes:

- `task`
- `inbox`
- `last_action_result`
- `remaining_steps`
- `warnings`
- `reward_breakdown`
- OpenEnv base fields: `reward`, `done`, and `metadata`

`SupportOpsState` includes:

- the full ticket workspace
- the current deterministic score
- criterion-level grading details
- action history
- the last typed reward object

## Reward design

Reward is dense across the full trajectory instead of only at the end.

- the main signal is score delta after each action
- every action carries a small time-cost penalty
- repeated actions incur a loop penalty
- invalid or destructive actions incur an additional penalty
- `finish` gives a small bonus when used at the right time and a penalty when used too early

This means agents get credit for correct intermediate work such as routing, tagging, or documenting the case, even before the task is fully complete.

## Tasks

The environment ships with three bundled tasks, each with a deterministic grader.

### 1. Easy: `easy_access_recovery`

Objective: handle a blocked SSO onboarding ticket.

Expected difficulty:

- one ticket
- straightforward routing
- simple customer follow-up

The agent must:

- set high priority
- route the ticket to Identity Support
- keep it waiting on the customer
- apply the right tags
- ask for the workspace URL, full email, and screenshot

### 2. Medium: `medium_refund_recovery`

Objective: process a valid annual renewal refund and downgrade request.

Expected difficulty:

- apply a billing policy correctly
- write both an internal note and a customer-facing reply
- resolve the case without over-escalation

The agent must:

- route the case to Billing Ops
- resolve it with an approved refund
- add an internal note citing the 7-day grace window and lack of usage
- explain refund timing and immediate Starter downgrade in the reply

### 3. Hard: `hard_export_incident`

Objective: coordinate a live CSV export incident across multiple tickets.

Expected difficulty:

- multi-ticket state tracking
- duplicate detection with constraints
- incident communication
- preservation of unrelated work

The agent must:

- escalate the VIP enterprise ticket to Engineering with urgent priority
- identify the same-company duplicate and merge it into the best parent ticket
- handle a separate affected account as part of the same incident without merging it
- communicate the workaround
- leave an unrelated billing ticket untouched

## Graders

Each task uses a deterministic weighted grader with scores in `[0.0, 1.0]`.

Criterion types include:

- exact field checks
- required tag checks
- reply and note keyword-group coverage
- duplicate merge checks
- untouched-record checks for unrelated work

This keeps scoring reproducible while still allowing partial credit for incomplete but correct progress.

## Project structure

```text
support_ops_env/
|-- README.md
|-- Dockerfile
|-- openenv.yaml
|-- pyproject.toml
|-- __init__.py
|-- baseline_openai.py
|-- client.py
|-- graders.py
|-- models.py
|-- smoke_test.py
|-- tasks.py
|-- trainer_env.py
|-- server/
|   |-- __init__.py
|   |-- app.py
|   |-- Dockerfile
|   |-- requirements.txt
|   `-- support_ops_env_environment.py
`-- tests/
    `-- test_support_ops_env.py
```

## Setup

All commands below assume you first move into the environment directory:

```bash
cd support_ops_env
```

### Local Python setup

```bash
cd support_ops_env
uv sync
```

### Run tests

```bash
cd support_ops_env
uv run --extra dev pytest
```

### Run the smoke test

This runs tests, local structure validation, and runtime validation against a live server:

```bash
cd support_ops_env
uv run smoke
```

### Validate structure with OpenEnv

```bash
cd support_ops_env
uvx --from openenv-core openenv validate .
```

### Run the server locally

```bash
cd support_ops_env
uv run server
```

Useful local routes:

- API docs: `http://localhost:8000/docs`
- health: `http://localhost:8000/health`
- schema: `http://localhost:8000/schema`
- metadata: `http://localhost:8000/metadata`

## Usage

### Local tuple-style loop

```python
from support_ops_env import SupportOpsAction, SupportOpsTrainerEnv
from support_ops_env.models import ActionType, Priority, Team, TicketStatus

env = SupportOpsTrainerEnv("easy_access_recovery")
obs = env.reset()

obs, reward, done, info = env.step(
    SupportOpsAction(
        action_type=ActionType.UPDATE_TICKET,
        ticket_id="T-1001",
        priority=Priority.HIGH,
        assigned_team=Team.IDENTITY_SUPPORT,
        status=TicketStatus.WAITING_ON_CUSTOMER,
        tags_to_add=["authentication", "sso"],
    )
)

state = env.state()
print(reward, done, state.current_score)
```

### OpenEnv client against a running server

```python
from support_ops_env import SupportOpsAction, SupportOpsEnv
from support_ops_env.models import ActionType

with SupportOpsEnv(base_url="http://localhost:8000") as env:
    result = env.reset(task_id="medium_refund_recovery")
    result = env.step(SupportOpsAction(action_type=ActionType.FINISH))
```

## Docker

Build and run from the environment root:

```bash
cd support_ops_env
docker build -t support-ops-env .
docker run -d --name openenv-support-ops -p 8000:7860 support-ops-env
```

Validate the running container:

```bash
cd support_ops_env
uvx --from openenv-core openenv validate http://127.0.0.1:8000
```

Stop and remove the container:

```bash
cd support_ops_env
docker stop openenv-support-ops
docker rm openenv-support-ops
```

The repository also keeps `server/Dockerfile` so `openenv build` and the scaffolded layout continue to work.

## Hugging Face Spaces deployment

This repository is structured as a Docker-based HF Space and tagged with `openenv` in the README front matter.

Recommended flow:

```bash
cd support_ops_env
uvx --from openenv-core openenv validate .
uvx --from openenv-core openenv push
```

Manual Docker Space flow:

1. Create a new Hugging Face Space with `sdk: docker`.
2. Push this directory as the Space root.
3. Keep the `openenv` tag in the README front matter.
4. The container will expose the OpenEnv server on port `8000`.
4. The container listens on port `7860`, which matches Hugging Face Docker Spaces.

## Baselines

### Scripted reference policy

These scores are deterministic and covered by the included tests:

| Task | Reference score |
| --- | --- |
| `easy_access_recovery` | `1.00` |
| `medium_refund_recovery` | `1.00` |
| `hard_export_incident` | `1.00` |

### OpenAI API baseline

The repository includes `baseline_openai.py`, which uses the OpenAI Python client and reads credentials from `OPENAI_API_KEY`.

Run it with:

```bash
cd support_ops_env
uv run baseline --model gpt-4.1-mini
```

Or:

```bash
cd support_ops_env
uv run python -m support_ops_env.baseline_openai --model gpt-4.1-mini
```

Output is a reproducible JSON summary over all three bundled tasks, including:

- per-task final score
- total reward
- step count
- full action trajectory

OpenAI scores are not embedded here because `OPENAI_API_KEY` was not configured during implementation. Once a key is available, run the baseline and add the resulting scores to this section.

## Verification status

The following checks passed locally:

- `uv run --extra dev pytest`
- `uvx --from openenv-core openenv validate .`
- `uvx --from openenv-core openenv validate http://127.0.0.1:8000`
- `docker build -t support-ops-env .`
- `docker run -d --name openenv-support-ops -p 8000:7860 support-ops-env`
- `uvx --from openenv-core openenv validate http://127.0.0.1:8000` against the running container
