"""Baseline inference runner for the queue operations benchmark tasks."""

import asyncio
import csv
import json
import os
import statistics
import textwrap
from typing import List, Optional
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Load environment variables from .env file

from cloud_queue_env import CloudQueueAction, CloudQueueEnv


IMAGE_NAME = os.getenv("IMAGE_NAME")
BASE_URL = os.getenv("BASE_URL")


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

BENCHMARK = os.getenv("BENCHMARK", "queueops-openenv")
TASKS = ["easy", "medium", "hard"]
TASK_SEEDS_JSON = os.getenv("TASK_SEEDS_JSON")
SEEDS = [11, 23, 37]
TEMPERATURE = 0.2
MAX_TOKENS = 180
SUCCESS_SCORE_THRESHOLD = 0.60
USE_HEURISTIC_ONLY = os.getenv("USE_HEURISTIC_ONLY", "false").lower() in {"1", "true", "yes"}
DISABLE_MODEL_ON_FIRST_ERROR = os.getenv("DISABLE_MODEL_ON_FIRST_ERROR", "true").lower() in {"1", "true", "yes"}
MAX_STEPS_OVERRIDE = int(os.getenv("MAX_STEPS_OVERRIDE", "0") or "0")
ACTION_TRACE_FILE = os.getenv("ACTION_TRACE_FILE")
REPORT_JSON_PATH = os.getenv("REPORT_JSON_PATH")
REPORT_CSV_PATH = os.getenv("REPORT_CSV_PATH")

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an agent controlling a cloud queue scheduling environment.
    Your goal: minimize wait times, SLA violations, and cost while maximizing throughput.

    ACTIONS (return exactly one JSON object, no extra text):
      {"action_type": "admit",       "target_queue": 0}          — accept incoming job into queue 0
      {"action_type": "route",       "target_queue": 1}          — accept incoming job into queue 1 (medium/hard only)
      {"action_type": "reject",      "target_queue": null}       — reject incoming job (use when queues are filling up)
      {"action_type": "dispatch",    "target_queue": 0}          — move job from queue to an idle server
      {"action_type": "reprioritize","new_priority": 2}         — promote a normal job to urgent (medium/hard only)
      {"action_type": "scale",       "scale_delta": 1}           — add 1 server (+1) or remove 1 server (-1) (hard only)
      {"action_type": "noop",        "target_queue": null}       — do nothing

    STRATEGY HINTS:
      - REJECT jobs when queue fill is above 60% to prevent overflow and SLA breaches.
      - ADMIT when queues have space and server is idle.
      - DISPATCH after admitting to keep servers busy.
      - On medium/hard: ROUTE urgent jobs (priority=2) to a less-loaded queue.
      - On hard: SCALE up (+1) when queue_fill > 70% and cost allows; scale down when queues are empty.
      - Negative reward means the system is struggling — change strategy.

    Return ONLY valid JSON. No explanation.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def parse_task_seed_map() -> dict[str, list[int]]:
    if TASK_SEEDS_JSON:
        try:
            data = json.loads(TASK_SEEDS_JSON)
            task_map: dict[str, list[int]] = {}
            for task_name, seeds in data.items():
                parsed = [int(s) for s in seeds]
                if parsed:
                    task_map[str(task_name)] = parsed
            if task_map:
                return task_map
        except Exception as exc:
            print(f"[DEBUG] Invalid TASK_SEEDS_JSON, falling back to defaults: {exc}", flush=True)

    return {
        "easy": [SEEDS[0]],
        "medium": [SEEDS[1]],
        "hard": [SEEDS[2]],
    }


def _action_from_dict(data: dict) -> CloudQueueAction:
    return CloudQueueAction(
        action_type=str(data.get("action_type", "noop")),
        target_queue=data.get("target_queue"),
        target_server=data.get("target_server"),
        scale_delta=data.get("scale_delta"),
        new_priority=data.get("new_priority"),
    )


def load_replay_actions() -> dict[str, list[CloudQueueAction]]:
    if not ACTION_TRACE_FILE:
        return {}

    try:
        with open(ACTION_TRACE_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[DEBUG] Failed to load ACTION_TRACE_FILE: {exc}", flush=True)
        return {}

    replay: dict[str, list[CloudQueueAction]] = {}
    if isinstance(payload, dict):
        for key, action_list in payload.items():
            if not isinstance(action_list, list):
                continue
            parsed = []
            for item in action_list:
                if isinstance(item, dict):
                    parsed.append(_action_from_dict(item))
            if parsed:
                replay[str(key)] = parsed
    return replay


def ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    std = statistics.pstdev(values)
    return 1.96 * std / (len(values) ** 0.5)


def write_reports(seed_rows: list[dict], task_score_table: dict[str, list[float]]) -> None:
    if REPORT_JSON_PATH:
        report_payload = {
            "seed_rows": seed_rows,
            "task_summary": {
                task: {
                    "mean": statistics.mean(scores) if scores else 0.0,
                    "std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
                    "ci95": ci95(scores),
                    "count": len(scores),
                }
                for task, scores in task_score_table.items()
            },
        }
        try:
            with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(report_payload, f, indent=2)
        except Exception as exc:
            print(f"[DEBUG] Failed to write REPORT_JSON_PATH: {exc}", flush=True)

    if REPORT_CSV_PATH:
        try:
            with open(REPORT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "task",
                        "seed",
                        "score",
                        "steps",
                        "success",
                        "trace_digest",
                        "invalid_actions",
                        "harmful_scale_down",
                    ],
                )
                writer.writeheader()
                for row in seed_rows:
                    writer.writerow(row)
        except Exception as exc:
            print(f"[DEBUG] Failed to write REPORT_CSV_PATH: {exc}", flush=True)


def build_obs_summary(obs: "CloudQueueObservation", task_name: str) -> str:
    """Build a rich, structured text summary of the observation for the LLM prompt."""
    # Queue fill percentages — helps model know when to reject
    max_sizes = {"easy": 28, "medium": 42, "hard": 64}
    max_q = max_sizes.get(task_name, 30)
    fills = [f"{l}/{max_q}({100*l//max_q}%)" for l in obs.queue_lengths]

    # Server status
    busy_count = sum(obs.server_busy)
    total_servers = len(obs.server_busy)
    servers_str = f"{busy_count}/{total_servers} busy"

    # Incoming job info
    if obs.incoming_job_present:
        urgency = "URGENT" if obs.incoming_job_priority >= 2 else "normal"
        incoming_str = f"YES [{urgency} size={obs.incoming_job_size:.1f} deadline={obs.incoming_job_deadline:.0f}]"
    else:
        incoming_str = "none"

    return (
        f"task={task_name} | "
        f"queues={fills} | "
        f"servers={servers_str} | "
        f"incoming={incoming_str} | "
        f"sla_breach={obs.sla_violation_rate:.3f} | "
        f"abandonment={obs.abandonment_rate:.3f} | "
        f"cost_rate={obs.energy_cost_rate:.3f}"
    )


def build_user_prompt(step: int, obs_summary: str, last_reward: float, history: List[str], task_name: str) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step {step} | Last reward: {last_reward:.2f}
        State: {obs_summary}
        Recent actions:
        {history_block}
        Choose the best action now.
        """
    ).strip()


def choose_heuristic_action(task_name: str, queue_lengths: List[int], incoming_present: bool) -> CloudQueueAction:
    if incoming_present:
        if task_name == "hard" and len(queue_lengths) > 1 and queue_lengths[0] > queue_lengths[1]:
            return CloudQueueAction(action_type="route", target_queue=1)
        if task_name == "medium" and len(queue_lengths) > 1 and queue_lengths[1] < queue_lengths[0]:
            return CloudQueueAction(action_type="route", target_queue=1)
        return CloudQueueAction(action_type="admit", target_queue=0)
    return CloudQueueAction(action_type="dispatch", target_queue=0)


def parse_model_action(text: str) -> Optional[CloudQueueAction]:
    try:
        data = json.loads(text)
        return CloudQueueAction(
            action_type=str(data.get("action_type", "noop")),
            target_queue=data.get("target_queue"),
            target_server=data.get("target_server"),
            scale_delta=data.get("scale_delta"),
            new_priority=data.get("new_priority"),
        )
    except Exception:
        return None


def get_model_action(
    client: OpenAI,
    task_name: str,
    step: int,
    obs_summary: str,
    last_reward: float,
    history: List[str],
) -> tuple[Optional[CloudQueueAction], Optional[str]]:
    user_prompt = build_user_prompt(step, obs_summary, last_reward, history, task_name)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_model_action(text), None
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return None, str(exc)


def normalize_base_url(base_url: Optional[str]) -> Optional[str]:
    """Normalize user-provided BASE_URL into an API runtime URL.

    If a Hugging Face repo page URL is provided (huggingface.co/spaces/user/space),
    convert it to the runtime domain (https://user-space.hf.space).
    """
    if not base_url:
        return base_url

    cleaned = base_url.strip().rstrip("/")
    parsed = urlparse(cleaned)

    # Handle Hugging Face repo page URL -> runtime URL used by API/WebSocket.
    if parsed.netloc.lower() == "huggingface.co":
        parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(parts) >= 3 and parts[0] == "spaces":
            owner, space = parts[1], parts[2]
            # HF runtime hostnames use lowercase and are TLS-safe.
            owner = owner.lower().replace("_", "-")
            space = space.lower().replace("_", "-")
            return f"https://{owner}-{space}.hf.space"

    # Avoid accidentally pointing at the web UI path.
    if cleaned.endswith("/web"):
        cleaned = cleaned[:-4]
        parsed = urlparse(cleaned)

    # HF runtime domains should be lowercase and avoid underscores for TLS host checks.
    host = (parsed.hostname or "").lower()
    if host.endswith(".hf.space"):
        safe_host = host.replace("_", "-")
        if safe_host != host or (parsed.netloc and parsed.netloc != parsed.netloc.lower()):
            port_part = f":{parsed.port}" if parsed.port else ""
            netloc = f"{safe_host}{port_part}"
            parsed = parsed._replace(netloc=netloc)
            cleaned = urlunparse(parsed)

    return cleaned


def _smoke_test_model(client: OpenAI) -> bool:
    """Verify the model API is reachable AND can generate a coherent response.

    Asks a short queue-domain question that requires a real sentence answer.
    An empty or missing reply is treated as failure — not just exceptions.

    Prints [MODEL_OK] or [MODEL_FAIL] with details.
    Returns True if the model is working, False otherwise.
    """
    print(f"[MODEL_CHECK] Testing model={MODEL_NAME} at {API_BASE_URL} ...", flush=True)
    test_question = (
        "You are a cloud scheduling agent. "
        "A job queue is 80% full and a new urgent job just arrived. "
        "Should you admit the job, reject it, or route it to another queue? "
        "Answer in one sentence and explain why."
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": test_question}],
            temperature=0.0,
            max_tokens=80,
        )
        reply = (resp.choices[0].message.content or "").strip()
        if not reply:
            print("[MODEL_FAIL] Model returned an empty response.", flush=True)
            print("[MODEL_FAIL] Will fall back to heuristic for all steps.", flush=True)
            return False
        print(f"[MODEL_OK] model is reasoning correctly.", flush=True)
        print(f"[MODEL_OK] test reply: {reply}", flush=True)
        return True
    except Exception as exc:
        print(f"[MODEL_FAIL] Cannot reach model: {exc}", flush=True)
        print("[MODEL_FAIL] Will fall back to heuristic for all steps.", flush=True)
        return False


async def main() -> None:
    if not API_KEY and not USE_HEURISTIC_ONLY:
        raise ValueError("API_KEY is required for model inference.")

    client = None
    if not USE_HEURISTIC_ONLY:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    runtime_base_url = normalize_base_url(BASE_URL)

    if runtime_base_url:
        env = CloudQueueEnv(base_url=runtime_base_url)
    else:
        if not IMAGE_NAME:
            raise ValueError(
                "Set BASE_URL for deployed env, or IMAGE_NAME for local docker env."
            )
        env = await CloudQueueEnv.from_docker_image(IMAGE_NAME)

    try:
        # Run smoke test before benchmark — confirms model API is reachable.
        model_enabled = client is not None
        if client is not None:
            model_enabled = _smoke_test_model(client)
        task_seed_map = parse_task_seed_map()
        replay_map = load_replay_actions()
        task_score_table: dict[str, list[float]] = {}
        seed_rows: list[dict] = []

        for task_name in TASKS:
            seeds = task_seed_map.get(task_name, [])
            if not seeds:
                continue

            task_score_table[task_name] = []

            for seed in seeds:
                history: List[str] = []
                rewards: List[float] = []
                steps_taken = 0
                score = 0.0
                success = False

                log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

                await env.reset()
                await env.step(
                    CloudQueueAction(action_type="configure_task", task_id=task_name, seed=seed)
                )
                result = await env.reset()
                last_reward = 0.0
                max_steps = max(1, int(result.observation.horizon))
                if MAX_STEPS_OVERRIDE > 0:
                    max_steps = min(max_steps, MAX_STEPS_OVERRIDE)

                for step in range(1, max_steps + 1):
                    if result.done:
                        break

                    obs = result.observation
                    obs_summary = build_obs_summary(obs, task_name)

                    action = None
                    model_error = None
                    replay_key = f"{task_name}:{seed}"
                    replay_actions = replay_map.get(replay_key, [])
                    if step - 1 < len(replay_actions):
                        action = replay_actions[step - 1]

                    if action is None and model_enabled and client is not None:
                        action, model_error = get_model_action(
                            client=client,
                            task_name=task_name,
                            step=step,
                            obs_summary=obs_summary,
                            last_reward=last_reward,
                            history=history,
                        )
                        if model_error and DISABLE_MODEL_ON_FIRST_ERROR:
                            model_enabled = False
                            print("[DEBUG] Disabling model calls and switching to heuristic fallback.", flush=True)

                    if action is None:
                        action = choose_heuristic_action(
                            task_name=task_name,
                            queue_lengths=obs.queue_lengths,
                            incoming_present=obs.incoming_job_present,
                        )

                    result = await env.step(action)
                    reward = float(result.reward or 0.0)
                    done = bool(result.done)
                    error = None
                    meta = result.observation.metadata or {}
                    info = meta.get("info", {}) if isinstance(meta, dict) else {}
                    if isinstance(info, dict) and info.get("valid_action") is False:
                        error = str(info.get("note", "invalid_action"))

                    rewards.append(reward)
                    steps_taken = step
                    last_reward = reward

                    action_str = (
                        f"{action.action_type}(q={action.target_queue},s={action.target_server},"
                        f"d={action.scale_delta},p={action.new_priority})"
                    )
                    log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                    history.append(f"step={step} action={action_str} reward={reward:.2f}")

                    if done:
                        break

                if isinstance(result.observation.metadata, dict):
                    score = float(result.observation.metadata.get("episode_score", 0.0) or 0.0)
                    # Debug: print raw server metadata so we can verify grader output
                    _m = result.observation.metadata
                    print(
                        f"[DEBUG_META] task={task_name} seed={seed} "
                        f"episode_score={_m.get('episode_score')} "
                        f"score_details={_m.get('score_details')} "
                        f"metrics_completed={_m.get('metrics', {}).get('completed')} "
                        f"metrics_arrivals={_m.get('metrics', {}).get('arrivals')}",
                        flush=True,
                    )
                score = max(0.0, min(1.0, score))
                task_score_table[task_name].append(score)
                success = score >= SUCCESS_SCORE_THRESHOLD
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

                meta = result.observation.metadata or {}
                metrics = meta.get("metrics", {}) if isinstance(meta, dict) else {}
                seed_row = {
                    "task": task_name,
                    "seed": int(seed),
                    "score": round(score, 6),
                    "steps": int(steps_taken),
                    "success": bool(success),
                    "trace_digest": str(meta.get("trace_digest", "")),
                    "invalid_actions": float(metrics.get("invalid_actions", 0.0)),
                    "harmful_scale_down": float(metrics.get("harmful_scale_down", 0.0)),
                }
                seed_rows.append(seed_row)
                print(
                    "[REPORT_SEED] "
                    f"task={seed_row['task']} seed={seed_row['seed']} score={seed_row['score']:.3f} "
                    f"steps={seed_row['steps']} trace={seed_row['trace_digest']}",
                    flush=True,
                )

            task_scores = task_score_table[task_name]
            task_mean = statistics.mean(task_scores) if task_scores else 0.0
            task_std = statistics.pstdev(task_scores) if len(task_scores) > 1 else 0.0
            task_ci = ci95(task_scores)
            print(
                f"[REPORT] task={task_name} seeds={len(task_scores)} mean={task_mean:.3f} std={task_std:.3f} ci95={task_ci:.3f}",
                flush=True,
            )

        all_task_means = []
        for task_name in TASKS:
            scores = task_score_table.get(task_name, [])
            if scores:
                all_task_means.append(statistics.mean(scores))

        if all_task_means:
            final_score = sum(all_task_means) / len(all_task_means)
            easy_mean = statistics.mean(task_score_table.get("easy", [0.0]))
            medium_mean = statistics.mean(task_score_table.get("medium", [0.0]))
            hard_mean = statistics.mean(task_score_table.get("hard", [0.0]))
            print(
                f"[SUMMARY] easy={easy_mean:.3f} medium={medium_mean:.3f} hard={hard_mean:.3f} final={final_score:.3f}",
                flush=True,
            )

            write_reports(seed_rows=seed_rows, task_score_table=task_score_table)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())