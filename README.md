---
title: Cloud Queue Env Environment Server
emoji: 🖨️
colorFrom: pink
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Cloud Queue Env Environment

A real-world queue-operations benchmark for OpenEnv.

This environment simulates service operations decisions humans make in production systems:
- Admission and rejection under load
- Queue routing and dispatching
- Priority handling for urgent traffic
- Capacity scaling under infrastructure cost constraints

The benchmark includes three deterministic tasks with partial graders in [0, 1]:
- easy: single-queue stability
- medium: multi-server priority routing
- hard: two-stage queue network with scaling

## Quick Start

Use the CloudQueueEnv client to connect to a running server or container:

```python
from cloud_queue_env import CloudQueueAction, CloudQueueEnv

try:
    env = CloudQueueEnv.from_docker_image("cloud_queue_env-env:latest")

    # Configure task + seed, then reset into that deterministic episode
    env.reset()
    env.step(CloudQueueAction(action_type="configure_task", task_id="easy", seed=11))
    result = env.reset()

    for _ in range(20):
        obs = result.observation
        if obs.incoming_job_present:
            action = CloudQueueAction(action_type="admit", target_queue=0)
        else:
            action = CloudQueueAction(action_type="dispatch", target_queue=0)

        result = env.step(action)
        print(
            f"step={obs.sim_time} queues={obs.queue_lengths} "
            f"reward={result.reward:.3f} done={result.done}"
        )
        if result.done:
            break

    final_score = result.observation.metadata.get("episode_score", 0.0)
    print(f"episode_score={final_score:.3f}")

finally:
    env.close()
```

The CloudQueueEnv.from_docker_image() method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t cloud_queue_env-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
CloudQueueAction fields:
- action_type: one of configure_task, admit, reject, route, dispatch, scale, reprioritize, noop
- target_queue: queue index for route/dispatch/admit
- target_server: optional server index
- scale_delta: server delta for scale action
- new_priority: new priority value for reprioritize
- task_id: easy/medium/hard (used with configure_task)
- seed: deterministic task seed (used with configure_task)

### Observation
CloudQueueObservation includes:
- task_id, sim_time, horizon
- queue_lengths, queue_wait_ema
- server_busy, server_remaining_service, utilization
- incoming_job_present, incoming_job_size, incoming_job_priority, incoming_job_deadline, incoming_job_type
- sla_violation_rate, abandonment_rate, throughput_recent, energy_cost_rate
- level, optional_history, action_mask
- reward, done, metadata

### Reward
Per-step reward is dense and multi-objective:

$$
r_t = 0.35R_{wait} + 0.20R_{throughput} + 0.20R_{sla} + 0.15R_{cost} + 0.05R_{fair} + 0.05R_{safe}
$$

Properties:
- Partial progress signal over the full trajectory
- Penalties for invalid actions and unsafe/noop behavior under congestion
- Bounded reward values for stability

### Deterministic Graders
Each task returns a deterministic episode_score in [0, 1], stored in observation metadata.

- easy score uses avg wait, throughput, rejection rate, and SLA violations
- medium score uses urgent/normal p95 waits, urgent SLA, throughput, and action cost
- hard score uses end-to-end p95, abandonment, SLA, throughput, infra cost, and fairness gap

If invalid action rate exceeds threshold, score is capped.

## Tasks

1. easy (single queue stability)
- one queue, one server
- objective: low wait with acceptable throughput and low rejection

2. medium (priority routing)
- two queues and multiple servers
- objective: protect urgent traffic while maintaining total performance

3. hard (queue network + scaling)
- two-stage queue network with bursty arrivals and heavy-tailed service times
- objective: balance latency/SLA/abandonment against infra cost and fairness

## Baseline Inference

Run baseline inference across easy/medium/hard:

```bash
API_KEY=your_provider_key python inference.py
```

Optional variables:
- API_KEY (OpenAI-compatible provider key for model calls)
- API_BASE_URL (default: https://router.huggingface.co/v1)
- MODEL_NAME (default: Qwen/Qwen2.5-72B-Instruct)
- BASE_URL (if using deployed space)
- IMAGE_NAME (if launching local docker image)
- USE_HEURISTIC_ONLY (true/false)
- DISABLE_MODEL_ON_FIRST_ERROR (true/false)
- MAX_STEPS_OVERRIDE (integer quick-test cap)
- TASK_SEEDS_JSON (JSON map for multi-seed runs)
- ACTION_TRACE_FILE (JSON replay file keyed by task:seed)
- REPORT_JSON_PATH (write seed/task report JSON)
- REPORT_CSV_PATH (write per-seed report CSV)

Output includes required line types:
- [START]
- [STEP]
- [END]

And final aggregate summary:
- [SUMMARY] easy=<...> medium=<...> hard=<...> final=<...>

V2 reporting also includes:
- [REPORT_SEED] task=<task_id> seed=<seed> score=<score> steps=<n> trace=<digest>
- [REPORT] task=<task_id> seeds=<n> mean=<score> std=<score> ci95=<score>

## Advanced Usage

### Connecting to an Existing Server

If you already have a Cloud Queue Env environment server running, you can connect directly:

```python
from cloud_queue_env import CloudQueueEnv

# Connect to existing server
cloud_queue_envenv = CloudQueueEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = cloud_queue_envenv.reset()
result = cloud_queue_envenv.step(CloudQueueAction(action_type="dispatch", target_queue=0))
```

Note: When connecting to an existing server, `cloud_queue_envenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from cloud_queue_env import CloudQueueAction, CloudQueueEnv

# Connect with context manager (auto-connects and closes)
with CloudQueueEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Initial queues: {result.observation.queue_lengths}")
    # Multiple steps with low latency
    for _ in range(10):
        result = env.step(CloudQueueAction(action_type="noop"))
        print(f"Reward: {result.reward:.3f}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    CloudQueueEnvironment,  # Pass class, not instance
    CloudQueueAction,
    CloudQueueObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from cloud_queue_env import CloudQueueAction, CloudQueueEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with CloudQueueEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(CloudQueueAction(action_type="dispatch", target_queue=i % 2))
        return client_id, result.observation.queue_lengths

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

Core files:
- models: typed action/observation schema
- server environment: queue simulation, reward shaping, grading
- inference script: task sweep and benchmark logging

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
cloud_queue_env/
├── .dockerignore
├── __init__.py
├── README.md
├── openenv.yaml
├── pyproject.toml
├── client.py
├── models.py
├── inference.py
├── IMPLEMENTATION_ROADMAP.md
└── server/
    ├── __init__.py
    ├── cloud_queue_env_environment.py
    ├── app.py
    └── Dockerfile
```
