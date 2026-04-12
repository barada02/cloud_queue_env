"""Strict model-only inference runner for the queue operations benchmark.

This variant intentionally removes heuristic fallback paths.
Every decision must come from either:
1) replay trace input (ACTION_TRACE_FILE), or
2) model output.

If model output is invalid/unavailable, the seed run is marked failed.
"""

import asyncio
import csv
import json
import os
import statistics
import textwrap
from typing import Any, List, Optional
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from cloud_queue_env import CloudQueueAction, CloudQueueEnv, CloudQueueObservation


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
MAX_TOKENS = 780
SUCCESS_SCORE_THRESHOLD = 0.60
# Test-friendly default. Set MAX_STEPS_OVERRIDE=0 for full horizon.
MAX_STEPS_OVERRIDE = int(os.getenv("MAX_STEPS_OVERRIDE", "8") or "8")
ACTION_TRACE_FILE = os.getenv("ACTION_TRACE_FILE")
REPORT_JSON_PATH = os.getenv("REPORT_JSON_PATH")
REPORT_CSV_PATH = os.getenv("REPORT_CSV_PATH")

OPEN_SCORE_MIN = 0.001
OPEN_SCORE_MAX = 0.999

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an agent controlling a cloud queue scheduling environment.
    Your goal: minimize wait times, SLA violations, and cost while maximizing throughput.

        OUTPUT FORMAT (strict):
        - Return exactly one JSON object.
        - No markdown, no code fences, no explanations, no extra keys.
        - Always include all fields below.

        Required JSON schema:
        {
            "action_type": "admit|reject|route|dispatch|scale|reprioritize|noop",
            "target_queue": integer or null,
            "target_server": integer or null,
            "scale_delta": integer or null,
            "new_priority": integer or null
        }

        Task constraints:
        - easy: only admit/reject/dispatch/noop
        - medium: only admit/reject/route/dispatch/reprioritize/noop
        - hard: only admit/reject/route/dispatch/reprioritize/scale/noop
    """
).strip()

ACTION_TYPES = (
    "configure_task",
    "admit",
    "reject",
    "route",
    "dispatch",
    "scale",
    "reprioritize",
    "noop",
)

TASK_ALLOWED_ACTIONS = {
    "easy": {"admit", "reject", "dispatch", "noop"},
    "medium": {"admit", "reject", "route", "dispatch", "reprioritize", "noop"},
    "hard": {"admit", "reject", "route", "dispatch", "reprioritize", "scale", "noop"},
}


def clamp_open_score(value: float) -> float:
    if not isinstance(value, (int, float)) or not (value == value):
        return OPEN_SCORE_MIN
    return max(OPEN_SCORE_MIN, min(OPEN_SCORE_MAX, float(value)))


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
                        "failure_reason",
                    ],
                )
                writer.writeheader()
                for row in seed_rows:
                    writer.writerow(row)
        except Exception as exc:
            print(f"[DEBUG] Failed to write REPORT_CSV_PATH: {exc}", flush=True)


def build_obs_summary(obs: CloudQueueObservation, task_name: str) -> str:
    max_sizes = {"easy": 28, "medium": 42, "hard": 64}
    max_q = max_sizes.get(task_name, 30)
    fills = [f"{l}/{max_q}({100*l//max_q}%)" for l in obs.queue_lengths]

    busy_count = sum(obs.server_busy)
    total_servers = len(obs.server_busy)
    servers_str = f"{busy_count}/{total_servers} busy"

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


def build_user_prompt(step: int, obs_summary: str, last_reward: float, history: List[str]) -> str:
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


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        txt = value.strip().lower()
        if txt in {"", "null", "none"}:
            return None
        try:
            return int(txt)
        except ValueError:
            try:
                return int(float(txt))
            except ValueError:
                return None
    return None


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    if cleaned.startswith("```"):
        chunks = [chunk.strip() for chunk in cleaned.split("```") if chunk.strip()]
        for chunk in chunks:
            candidate = chunk
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    return parsed[0]
            except Exception:
                continue

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
    except Exception:
        pass

    start = 0
    while True:
        open_idx = cleaned.find("{", start)
        if open_idx < 0:
            return None
        depth = 0
        for i in range(open_idx, len(cleaned)):
            ch = cleaned[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[open_idx : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        break
        start = open_idx + 1


def _normalize_action_payload(data: dict[str, Any], task_name: str) -> Optional[dict[str, Any]]:
    action_type = str(data.get("action_type", "noop")).strip().lower()
    if action_type not in ACTION_TYPES:
        return None
    if action_type not in TASK_ALLOWED_ACTIONS.get(task_name, set(ACTION_TYPES)):
        return None

    target_queue = _coerce_optional_int(data.get("target_queue"))
    target_server = _coerce_optional_int(data.get("target_server"))
    scale_delta = _coerce_optional_int(data.get("scale_delta"))
    new_priority = _coerce_optional_int(data.get("new_priority"))

    if action_type in {"admit", "route", "dispatch"} and target_queue is None:
        target_queue = 0
    if action_type in {"reject", "noop"}:
        target_queue = None
        target_server = None

    if action_type == "scale":
        if scale_delta is None:
            return None
        scale_delta = max(-2, min(2, scale_delta))
    else:
        scale_delta = None

    if action_type == "reprioritize":
        if new_priority is None:
            new_priority = 2
    else:
        new_priority = None

    return {
        "action_type": action_type,
        "target_queue": target_queue,
        "target_server": target_server,
        "scale_delta": scale_delta,
        "new_priority": new_priority,
    }


def parse_model_action(text: str, task_name: str) -> Optional[CloudQueueAction]:
    data = _extract_json_object(text)
    if data is None:
        return None
    payload = _normalize_action_payload(data, task_name)
    if payload is None:
        return None
    try:
        return CloudQueueAction(**payload)
    except Exception:
        return None


def _single_line(text: str) -> str:
    return " ".join((text or "").split())


def get_model_action(
    client: OpenAI,
    task_name: str,
    step: int,
    obs_summary: str,
    last_reward: float,
    history: List[str],
) -> tuple[Optional[CloudQueueAction], Optional[str]]:
    user_prompt = build_user_prompt(step, obs_summary, last_reward, history)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )

        text = (completion.choices[0].message.content or "").strip()
        print(
            f"[MODEL_OUTPUT] task={task_name} step={step} raw={_single_line(text)}",
            flush=True,
        )
        action = parse_model_action(text, task_name)
        if action is None:
            preview = " ".join(text.split())[:180]
            return None, f"invalid_model_action_payload: {preview}"
        return action, None
    except Exception as exc:
        return None, str(exc)


def get_model_action_with_retry(
    client: OpenAI,
    task_name: str,
    step: int,
    obs_summary: str,
    last_reward: float,
    history: List[str],
    retries: int = 2,
) -> tuple[Optional[CloudQueueAction], Optional[str]]:
    last_error: Optional[str] = None
    for attempt in range(1, retries + 2):
        action, error = get_model_action(
            client=client,
            task_name=task_name,
            step=step,
            obs_summary=obs_summary,
            last_reward=last_reward,
            history=history,
        )
        if action is not None:
            return action, None
        last_error = error
        print(f"[DEBUG] Model action parse failed on attempt={attempt}: {error}", flush=True)
    return None, last_error


def normalize_base_url(base_url: Optional[str]) -> Optional[str]:
    if not base_url:
        return base_url

    cleaned = base_url.strip().rstrip("/")
    parsed = urlparse(cleaned)

    if parsed.netloc.lower() == "huggingface.co":
        parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(parts) >= 3 and parts[0] == "spaces":
            owner, space = parts[1], parts[2]
            owner = owner.lower().replace("_", "-")
            space = space.lower().replace("_", "-")
            return f"https://{owner}-{space}.hf.space"

    if cleaned.endswith("/web"):
        cleaned = cleaned[:-4]
        parsed = urlparse(cleaned)

    host = (parsed.hostname or "").lower()
    if host.endswith(".hf.space"):
        safe_host = host.replace("_", "-")
        if safe_host != host or (parsed.netloc and parsed.netloc != parsed.netloc.lower()):
            port_part = f":{parsed.port}" if parsed.port else ""
            parsed = parsed._replace(netloc=f"{safe_host}{port_part}")
            cleaned = urlunparse(parsed)

    return cleaned


def _smoke_test_model(client: OpenAI) -> bool:
    print(f"[MODEL_CHECK] Testing model={MODEL_NAME} at {API_BASE_URL} ...", flush=True)
    test_question = (
        "You are a cloud scheduling agent. "
        "A job queue is 80% full and a new urgent job just arrived. "
        "Should you admit the job, reject it, or route it to another queue? "
        "Answer with exactly one JSON object containing action_type and optional fields."
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
            return False
        print("[MODEL_OK] model endpoint reachable.", flush=True)
        return True
    except Exception as exc:
        print(f"[MODEL_FAIL] Cannot reach model: {exc}", flush=True)
        return False


async def main() -> None:
    if not API_KEY:
        raise ValueError("API_KEY or HF_TOKEN is required for strict model inference.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    if not _smoke_test_model(client):
        raise RuntimeError("Model smoke test failed. Aborting strict model-only run.")

    runtime_base_url = normalize_base_url(BASE_URL)
    if runtime_base_url:
        env = CloudQueueEnv(base_url=runtime_base_url)
    else:
        if not IMAGE_NAME:
            raise ValueError("Set BASE_URL for deployed env, or IMAGE_NAME for local docker env.")
        env = await CloudQueueEnv.from_docker_image(IMAGE_NAME)

    try:
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
                score = OPEN_SCORE_MIN
                success = False
                failure_reason: Optional[str] = None

                log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

                await env.reset()
                await env.step(CloudQueueAction(action_type="configure_task", task_id=task_name, seed=seed))
                result = await env.reset()

                last_reward = 0.0
                max_steps = max(1, int(result.observation.horizon))
                if MAX_STEPS_OVERRIDE > 0:
                    max_steps = min(max_steps, MAX_STEPS_OVERRIDE)

                replay_key = f"{task_name}:{seed}"
                replay_actions = replay_map.get(replay_key, [])

                for step in range(1, max_steps + 1):
                    if result.done:
                        break

                    obs = result.observation
                    obs_summary = build_obs_summary(obs, task_name)

                    action: Optional[CloudQueueAction] = None
                    model_error: Optional[str] = None

                    if step - 1 < len(replay_actions):
                        action = replay_actions[step - 1]
                    else:
                        action, model_error = get_model_action_with_retry(
                            client=client,
                            task_name=task_name,
                            step=step,
                            obs_summary=obs_summary,
                            last_reward=last_reward,
                            history=history,
                            retries=2,
                        )

                    if action is None:
                        failure_reason = f"model_action_unavailable: {model_error}"
                        log_step(
                            step=step,
                            action="model_action_error",
                            reward=0.0,
                            done=True,
                            error=failure_reason,
                        )
                        steps_taken = step
                        break

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

                if failure_reason is None and isinstance(result.observation.metadata, dict):
                    score = float(result.observation.metadata.get("episode_score", OPEN_SCORE_MIN) or OPEN_SCORE_MIN)
                    _m = result.observation.metadata
                    print(
                        f"[DEBUG_META] task={task_name} seed={seed} "
                        f"episode_score={_m.get('episode_score')} "
                        f"score_details={_m.get('score_details')} "
                        f"metrics_completed={_m.get('metrics', {}).get('completed')} "
                        f"metrics_arrivals={_m.get('metrics', {}).get('arrivals')}",
                        flush=True,
                    )
                elif failure_reason is not None:
                    score = OPEN_SCORE_MIN

                if failure_reason is None and not bool(result.done):
                    failure_reason = "episode_not_done_within_max_steps"
                    print(
                        "[DEBUG] Episode ended early before done=true; "
                        "set MAX_STEPS_OVERRIDE=0 or unset it for valid benchmark scores.",
                        flush=True,
                    )
                    score = OPEN_SCORE_MIN

                score = clamp_open_score(score)
                task_score_table[task_name].append(score)
                success = failure_reason is None and score >= SUCCESS_SCORE_THRESHOLD
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

                meta = result.observation.metadata or {}
                metrics = meta.get("metrics", {}) if isinstance(meta, dict) else {}
                seed_row = {
                    "task": task_name,
                    "seed": int(seed),
                    "score": score,
                    "steps": int(steps_taken),
                    "success": bool(success),
                    "trace_digest": str(meta.get("trace_digest", "")),
                    "invalid_actions": float(metrics.get("invalid_actions", 0.0)),
                    "harmful_scale_down": float(metrics.get("harmful_scale_down", 0.0)),
                    "failure_reason": failure_reason or "",
                }
                seed_rows.append(seed_row)
                print(
                    "[REPORT_SEED] "
                    f"task={seed_row['task']} seed={seed_row['seed']} score={seed_row['score']:.3f} "
                    f"steps={seed_row['steps']} trace={seed_row['trace_digest']}",
                    flush=True,
                )

            task_scores = task_score_table[task_name]
            task_mean = statistics.mean(task_scores) if task_scores else OPEN_SCORE_MIN
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
            final_score = clamp_open_score(sum(all_task_means) / len(all_task_means))
            easy_mean = clamp_open_score(statistics.mean(task_score_table.get("easy", [OPEN_SCORE_MIN])))
            medium_mean = clamp_open_score(statistics.mean(task_score_table.get("medium", [OPEN_SCORE_MIN])))
            hard_mean = clamp_open_score(statistics.mean(task_score_table.get("hard", [OPEN_SCORE_MIN])))
            print(
                f"[SUMMARY] easy={easy_mean:.3f} medium={medium_mean:.3f} hard={hard_mean:.3f} final={final_score:.3f}",
                flush=True,
            )

            write_reports(seed_rows=seed_rows, task_score_table=task_score_table)

    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error (container cleanup): {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
