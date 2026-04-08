"""Baseline inference runner for the queue operations benchmark tasks."""

import asyncio
import json
import os
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

API_KEY = os.getenv("API_KEY")

BENCHMARK = os.getenv("BENCHMARK", "queueops-openenv")
TASKS = ["easy", "medium", "hard"]
SEEDS = [11, 23, 37]
TEMPERATURE = 0.2
MAX_TOKENS = 180
SUCCESS_SCORE_THRESHOLD = 0.60
USE_HEURISTIC_ONLY = os.getenv("USE_HEURISTIC_ONLY", "false").lower() in {"1", "true", "yes"}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are controlling a queue operations environment.
    Return exactly one JSON object with keys:
    action_type, target_queue, target_server, scale_delta, new_priority.
    Allowed action_type values: admit, reject, route, dispatch, scale, reprioritize, noop.
    Keep values simple integers or null when not used.
    Return only JSON and no extra text.
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


def build_user_prompt(step: int, obs_summary: str, last_reward: float, history: List[str], task_name: str) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Task: {task_name}
        Step: {step}
        Observation summary: {obs_summary}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Decide the next action.
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
) -> Optional[CloudQueueAction]:
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
        return parse_model_action(text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return None


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


async def main() -> None:
    if not API_KEY and not USE_HEURISTIC_ONLY:
        raise ValueError("HF_TOKEN is required for model inference.")

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
        task_scores: List[float] = []

        for task_name, seed in zip(TASKS, SEEDS):
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

            for step in range(1, max_steps + 1):
                if result.done:
                    break

                obs = result.observation
                obs_summary = (
                    f"queues={obs.queue_lengths}, incoming={obs.incoming_job_present}, "
                    f"priority={obs.incoming_job_priority}, sla_rate={obs.sla_violation_rate:.3f}, "
                    f"abandonment={obs.abandonment_rate:.3f}, cost={obs.energy_cost_rate:.3f}"
                )

                action = None
                if client is not None:
                    action = get_model_action(
                        client=client,
                        task_name=task_name,
                        step=step,
                        obs_summary=obs_summary,
                        last_reward=last_reward,
                        history=history,
                    )
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
            score = max(0.0, min(1.0, score))
            task_scores.append(score)
            success = score >= SUCCESS_SCORE_THRESHOLD
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

        if task_scores:
            final_score = sum(task_scores) / len(task_scores)
            print(
                f"[SUMMARY] easy={task_scores[0]:.3f} medium={task_scores[1]:.3f} hard={task_scores[2]:.3f} final={final_score:.3f}",
                flush=True,
            )

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())