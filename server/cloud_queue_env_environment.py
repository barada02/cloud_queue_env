# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Queue operations environment with deterministic task grading."""

import math
import random
import hashlib
from collections import deque
from dataclasses import dataclass
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CloudQueueAction, CloudQueueObservation
except ImportError:
    from models import CloudQueueAction, CloudQueueObservation


@dataclass
class TaskConfig:
    task_id: str
    horizon: int
    level: float
    queue_count: int
    initial_servers: int
    min_servers: int
    max_servers: int
    arrival_rate: float
    urgent_ratio: float
    service_mean: float
    deadline_base: int
    allow_scaling: bool
    allow_priority: bool
    two_stage: bool
    server_cost: float
    max_queue_size: int
    score_refs: dict[str, float]


class CloudQueueEnvironment(Environment):
    """Deterministic queueing environment with easy/medium/hard benchmark tasks."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._task_configs = self._build_task_configs()
        self._active_task_id = "easy"
        self._pending_task_id = "easy"
        self._pending_seed = 7
        self._rng_streams: dict[str, random.Random] = {}
        self._rng_stream_seeds: dict[str, int] = {}
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._sim_time = 0
        self._queues: list[deque[dict]] = []
        self._servers: list[dict] = []
        self._incoming_job: dict | None = None
        self._done = False
        self._wait_ema: list[float] = []
        self._utilization_ema: list[float] = []
        self._metrics: dict[str, float] = {}
        self._recent_rewards: deque[float] = deque(maxlen=8)
        self._action_trace: list[str] = []
        self._reset_runtime_state()

    def _build_task_configs(self) -> dict[str, TaskConfig]:
        return {
            "easy": TaskConfig(
                task_id="easy",
                horizon=150,
                level=1.0,
                queue_count=1,
                initial_servers=1,
                min_servers=1,
                max_servers=1,
                arrival_rate=0.78,
                urgent_ratio=0.0,
                service_mean=1.6,
                deadline_base=10,
                allow_scaling=False,
                allow_priority=False,
                two_stage=False,
                server_cost=0.04,
                max_queue_size=28,
                score_refs={"wait": 6.0, "thr": 70.0, "rej": 0.3, "sla": 0.3},
            ),
            "medium": TaskConfig(
                task_id="medium",
                horizon=200,
                level=2.3,
                queue_count=2,
                initial_servers=3,
                min_servers=2,
                max_servers=4,
                arrival_rate=1.15,
                urgent_ratio=0.28,
                service_mean=1.8,
                deadline_base=8,
                allow_scaling=False,
                allow_priority=True,
                two_stage=False,
                server_cost=0.06,
                max_queue_size=42,
                score_refs={"uw": 7.0, "nw": 10.0, "usla": 0.25, "thr": 125.0, "cost": 14.0},
            ),
            "hard": TaskConfig(
                task_id="hard",
                horizon=250,
                level=4.0,
                queue_count=2,
                initial_servers=3,
                min_servers=1,
                max_servers=6,
                arrival_rate=1.45,
                urgent_ratio=0.35,
                service_mean=2.2,
                deadline_base=7,
                allow_scaling=True,
                allow_priority=True,
                two_stage=True,
                server_cost=0.1,
                max_queue_size=64,
                score_refs={
                    "e2e": 14.0,
                    "abd": 0.25,
                    "sla": 0.3,
                    "thr": 145.0,
                    "cost": 28.0,
                    "fair": 0.35,
                },
            ),
        }

    def _reset_runtime_state(self) -> None:
        cfg = self._task_configs[self._active_task_id]
        self._sim_time = 0
        self._done = False
        self._incoming_job = None
        self._action_trace = []
        self._queues = [deque() for _ in range(cfg.queue_count)]
        self._servers = [
            {"remaining": 0.0, "job": None, "active": True}
            for _ in range(cfg.initial_servers)
        ]
        self._wait_ema = [0.0 for _ in range(cfg.queue_count)]
        self._utilization_ema = [0.0 for _ in range(cfg.max_servers)]
        self._recent_rewards.clear()
        self._metrics = {
            "arrivals": 0.0,
            "accepted": 0.0,
            "rejected": 0.0,
            "completed": 0.0,
            "completed_urgent": 0.0,
            "abandoned": 0.0,
            "wait_sum": 0.0,
            "wait_count": 0.0,
            "wait_sum_urgent": 0.0,
            "wait_count_urgent": 0.0,
            "wait_sum_normal": 0.0,
            "wait_count_normal": 0.0,
            "sla_breaches": 0.0,
            "sla_breaches_urgent": 0.0,
            "invalid_actions": 0.0,
            "noop_under_load": 0.0,
            "harmful_scale_down": 0.0,
            "action_cost": 0.0,
            "infra_cost": 0.0,
            "fairness_gap_sum": 0.0,
            "fairness_gap_count": 0.0,
        }
        self._wait_samples_all: list[float] = []
        self._wait_samples_urgent: list[float] = []
        self._wait_samples_normal: list[float] = []
        self._e2e_wait_samples: list[float] = []

    def _init_rng_streams(self, base_seed: int) -> None:
        self._rng_stream_seeds = {
            "arrivals": int(base_seed) + 101,
            "service": int(base_seed) + 211,
            "abandonment": int(base_seed) + 307,
            "exogenous": int(base_seed) + 401,
        }
        self._rng_streams = {
            key: random.Random(seed) for key, seed in self._rng_stream_seeds.items()
        }

    def _rng(self, stream: str) -> random.Random:
        return self._rng_streams[stream]

    def _sample_poisson(self, lam: float, rng: random.Random) -> int:
        lam = max(0.0, lam)
        if lam == 0.0:
            return 0
        # Knuth algorithm is sufficient for this environment's lambda scale.
        l_term = math.exp(-lam)
        k = 0
        p = 1.0
        while p > l_term:
            k += 1
            p *= rng.random()
        return max(0, k - 1)

    def _trace_digest(self) -> str:
        raw = f"task={self._active_task_id}|seed={self._pending_seed}|" + "|".join(self._action_trace)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def reset(self) -> CloudQueueObservation:
        self._active_task_id = self._pending_task_id if self._pending_task_id in self._task_configs else "easy"
        self._init_rng_streams(self._pending_seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_runtime_state()
        return self._build_observation(reward=0.0, done=False, info={"event": "reset"})

    def _clamp(self, value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _sample_service_time(self, cfg: TaskConfig) -> float:
        service_rng = self._rng("service")
        if cfg.task_id == "hard":
            heavy = service_rng.random() < 0.22
            if heavy:
                return self._clamp(service_rng.lognormvariate(1.2, 0.7), 1.0, 12.0)
        return self._clamp(service_rng.expovariate(1.0 / cfg.service_mean), 0.5, 10.0)

    def _sample_arrivals(self, cfg: TaskConfig) -> int:
        arrival_rng = self._rng("arrivals")
        exogenous_rng = self._rng("exogenous")
        rate = cfg.arrival_rate
        if cfg.task_id == "hard":
            wave = 0.35 * math.sin((self._sim_time + 1) / 13.0)
            jitter = exogenous_rng.uniform(-0.05, 0.05)
            rate += wave + jitter
        return self._sample_poisson(rate, arrival_rng)

    def _spawn_incoming_job(self, cfg: TaskConfig) -> None:
        arrivals = self._sample_arrivals(cfg)
        if arrivals <= 0:
            self._incoming_job = None
            return
        arrival_rng = self._rng("arrivals")
        priority = 2 if arrival_rng.random() < cfg.urgent_ratio else 1
        size = self._sample_service_time(cfg)
        self._incoming_job = {
            "priority": priority,
            "queue": 0,
            "created_step": self._state.step_count,
            "wait": 0.0,
            "size": size,
            "remaining": size,
            "deadline": self._state.step_count + cfg.deadline_base - (1 if priority == 2 else 0),
            "type": 1 if priority == 2 else 0,
            "stage": 0,
        }
        self._metrics["arrivals"] += 1.0

    def _update_wait_and_abandonment(self, cfg: TaskConfig) -> float:
        abandonment_rng = self._rng("abandonment")
        abandoned_this_step = 0.0
        for qi, q in enumerate(self._queues):
            kept: deque[dict] = deque()
            while q:
                job = q.popleft()
                job["wait"] += 1.0
                patience = cfg.deadline_base + (2 if job["priority"] == 2 else 4)
                if cfg.task_id == "hard" and job["wait"] > patience and abandonment_rng.random() < 0.35:
                    abandoned_this_step += 1.0
                    continue
                kept.append(job)
            self._queues[qi] = kept
        if abandoned_this_step:
            self._metrics["abandoned"] += abandoned_this_step
        return abandoned_this_step

    def _complete_job(self, cfg: TaskConfig, job: dict) -> None:
        if cfg.two_stage and job["stage"] == 0:
            forwarded = dict(job)
            forwarded["stage"] = 1
            forwarded["queue"] = min(1, len(self._queues) - 1)
            forwarded["remaining"] = self._sample_service_time(cfg)
            self._queues[forwarded["queue"]].append(forwarded)
            return

        self._metrics["completed"] += 1.0
        wait = float(self._state.step_count - job["created_step"])
        self._metrics["wait_sum"] += wait
        self._metrics["wait_count"] += 1.0
        self._wait_samples_all.append(wait)
        self._e2e_wait_samples.append(wait)
        if job["priority"] == 2:
            self._metrics["completed_urgent"] += 1.0
            self._metrics["wait_sum_urgent"] += wait
            self._metrics["wait_count_urgent"] += 1.0
            self._wait_samples_urgent.append(wait)
        else:
            self._metrics["wait_sum_normal"] += wait
            self._metrics["wait_count_normal"] += 1.0
            self._wait_samples_normal.append(wait)
        if self._state.step_count > job["deadline"]:
            self._metrics["sla_breaches"] += 1.0
            if job["priority"] == 2:
                self._metrics["sla_breaches_urgent"] += 1.0

    def _process_servers(self, cfg: TaskConfig) -> float:
        completed_this_step = 0.0
        for si, server in enumerate(self._servers):
            if not server["active"]:
                continue
            if server["remaining"] > 0:
                server["remaining"] = max(0.0, server["remaining"] - 1.0)
            if server["remaining"] <= 0 and server["job"] is not None:
                self._complete_job(cfg, server["job"])
                completed_this_step += 1.0
                server["job"] = None
            busy_flag = 1.0 if server["job"] is not None else 0.0
            if si < len(self._utilization_ema):
                self._utilization_ema[si] = 0.9 * self._utilization_ema[si] + 0.1 * busy_flag
        return completed_this_step

    def _admit_job(self, cfg: TaskConfig, queue_idx: int) -> tuple[bool, str]:
        if self._incoming_job is None:
            return False, "no_incoming_job"
        if queue_idx < 0 or queue_idx >= len(self._queues):
            return False, "invalid_queue"
        if len(self._queues[queue_idx]) >= cfg.max_queue_size:
            self._metrics["rejected"] += 1.0
            self._incoming_job = None
            return True, "queue_full_rejected"
        job = dict(self._incoming_job)
        job["queue"] = queue_idx
        self._queues[queue_idx].append(job)
        self._incoming_job = None
        self._metrics["accepted"] += 1.0
        return True, "admitted"

    def _dispatch(self, queue_idx: int | None) -> tuple[bool, str]:
        target = 0 if queue_idx is None else queue_idx
        if target < 0 or target >= len(self._queues):
            return False, "invalid_dispatch_queue"
        for server in self._servers:
            if not server["active"]:
                continue
            if server["job"] is None and self._queues[target]:
                server["job"] = self._queues[target].popleft()
                server["remaining"] = server["job"]["remaining"]
                return True, "dispatched"
        return False, "no_idle_server_or_empty_queue"

    def _autodispatch(self) -> None:
        for server in self._servers:
            if not server["active"] or server["job"] is not None:
                continue
            for q in self._queues:
                if q:
                    server["job"] = q.popleft()
                    server["remaining"] = server["job"]["remaining"]
                    break

    def _apply_action(self, action: CloudQueueAction, cfg: TaskConfig) -> tuple[bool, str]:
        action_type = (action.action_type or "noop").lower()

        if action_type == "configure_task":
            if action.task_id and action.task_id in self._task_configs:
                self._pending_task_id = action.task_id
            if action.seed is not None:
                self._pending_seed = int(action.seed)
            return True, "configuration_updated_for_next_reset"

        if self._done:
            return False, "episode_already_done"

        if action_type == "admit":
            queue_idx = action.target_queue if action.target_queue is not None else 0
            return self._admit_job(cfg, queue_idx)

        if action_type == "reject":
            if self._incoming_job is None:
                return False, "no_incoming_job"
            self._incoming_job = None
            self._metrics["rejected"] += 1.0
            return True, "rejected"

        if action_type == "route":
            queue_idx = action.target_queue if action.target_queue is not None else 0
            return self._admit_job(cfg, queue_idx)

        if action_type == "dispatch":
            return self._dispatch(action.target_queue)

        if action_type == "scale":
            if not cfg.allow_scaling:
                return False, "scaling_not_supported_for_task"
            delta = action.scale_delta if action.scale_delta is not None else 0
            if delta == 0:
                return True, "no_scale_change"
            active_count = sum(1 for s in self._servers if s["active"])
            requested = int(self._clamp(active_count + delta, cfg.min_servers, cfg.max_servers))
            if requested == active_count:
                return True, "scale_clamped_no_change"
            if requested > active_count:
                for _ in range(requested - active_count):
                    self._servers.append({"remaining": 0.0, "job": None, "active": True})
                    self._utilization_ema.append(0.0)
            else:
                to_disable = active_count - requested
                for server in reversed(self._servers):
                    if to_disable == 0:
                        break
                    if server["active"] and server["job"] is None:
                        server["active"] = False
                        to_disable -= 1
            self._metrics["action_cost"] += abs(delta) * 0.35
            return True, "scaled"

        if action_type == "reprioritize":
            if not cfg.allow_priority:
                return False, "reprioritize_not_supported_for_task"
            new_priority = 2 if (action.new_priority or 1) >= 2 else 1
            for q in self._queues:
                for job in q:
                    if job["priority"] == 1:
                        job["priority"] = new_priority
                        return True, "reprioritized"
            return False, "no_eligible_job"

        if action_type == "noop":
            return True, "noop"

        return False, "unknown_action_type"

    def _percentile(self, values: list[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = int(self._clamp(round((len(ordered) - 1) * p), 0, len(ordered) - 1))
        return float(ordered[idx])

    def _safe_div(self, numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    def _current_fairness_gap(self) -> float:
        urgent_avg = self._safe_div(self._metrics["wait_sum_urgent"], self._metrics["wait_count_urgent"])
        normal_avg = self._safe_div(self._metrics["wait_sum_normal"], self._metrics["wait_count_normal"])
        scale = max(1.0, urgent_avg + normal_avg)
        return abs(urgent_avg - normal_avg) / scale

    def _compute_reward(
        self,
        cfg: TaskConfig,
        action_ok: bool,
        action_type: str,
        action_scale_delta: int,
        completed_step: float,
    ) -> tuple[float, dict[str, float]]:
        avg_wait = self._safe_div(self._metrics["wait_sum"], self._metrics["wait_count"])
        queue_pressure = sum(len(q) for q in self._queues) / max(1.0, float(cfg.max_queue_size))
        r_wait = -self._clamp(avg_wait / max(cfg.deadline_base, 1), 0.0, 1.5) - 0.15 * self._clamp(queue_pressure, 0.0, 1.5)
        r_throughput = self._clamp(completed_step / max(1.0, float(cfg.initial_servers)), 0.0, 1.0)
        total_decisions = max(1.0, self._metrics["completed"] + self._metrics["abandoned"])
        r_sla = -self._clamp(self._metrics["sla_breaches"] / total_decisions, 0.0, 1.0)
        active_servers = sum(1 for s in self._servers if s["active"])
        r_cost = -self._clamp(active_servers / max(1.0, float(cfg.max_servers)), 0.0, 1.0)
        fairness_gap = self._current_fairness_gap()
        r_fair = -self._clamp(fairness_gap / 0.5, 0.0, 1.0)
        r_safe = 0.0 if action_ok else -1.0
        if not action_ok:
            self._metrics["invalid_actions"] += 1.0
        if action_type == "noop" and self._incoming_job is not None and sum(len(q) for q in self._queues) > 0:
            r_safe -= 0.05
            self._metrics["noop_under_load"] += 1.0

        arrivals = max(1.0, self._metrics["arrivals"])
        rejection_rate = self._safe_div(self._metrics["rejected"], arrivals)
        if arrivals > 10 and rejection_rate > 0.4:
            r_safe -= self._clamp((rejection_rate - 0.4) * 0.4, 0.0, 0.2)

        if action_type == "scale" and action_scale_delta < 0 and queue_pressure > 0.45:
            overload_penalty = self._clamp((queue_pressure - 0.45) * 0.5, 0.0, 0.25)
            r_safe -= overload_penalty
            self._metrics["harmful_scale_down"] += 1.0

        reward = 0.35 * r_wait + 0.20 * r_throughput + 0.20 * r_sla + 0.15 * r_cost + 0.05 * r_fair + 0.05 * r_safe
        reward = self._clamp(reward, -1.0, 1.0)
        self._recent_rewards.append(reward)

        self._metrics["infra_cost"] += active_servers * cfg.server_cost
        self._metrics["fairness_gap_sum"] += fairness_gap
        self._metrics["fairness_gap_count"] += 1.0

        components = {
            "wait": round(r_wait, 4),
            "throughput": round(r_throughput, 4),
            "sla": round(r_sla, 4),
            "cost": round(r_cost, 4),
            "fairness": round(r_fair, 4),
            "safety": round(r_safe, 4),
        }
        return reward, components

    def _score_task(self, cfg: TaskConfig) -> tuple[float, dict[str, float]]:
        # c01: clamp individual sub-score components to [0, 1] inclusive.
        def c01(value: float) -> float:
            if not math.isfinite(value):
                return 0.0
            return self._clamp(value, 0.0, 1.0)

        # _strict01: final clamp applied only to the episode score.
        # Validator requires score strictly in (0, 1) — never 0.0 or 1.0.
        _SCORE_MIN = 0.001
        _SCORE_MAX = 0.999

        def strict01(value: float) -> float:
            if not math.isfinite(value):
                return _SCORE_MIN
            return self._clamp(value, _SCORE_MIN, _SCORE_MAX)

        completed = self._metrics["completed"]
        arrivals = self._metrics["arrivals"]
        rejected = self._metrics["rejected"]
        avg_wait = self._safe_div(self._metrics["wait_sum"], self._metrics["wait_count"])
        rejection_rate = self._safe_div(rejected, arrivals)
        sla_rate = self._safe_div(self._metrics["sla_breaches"], max(1.0, completed))
        throughput = completed
        fairness_gap = self._safe_div(self._metrics["fairness_gap_sum"], self._metrics["fairness_gap_count"])

        if cfg.task_id == "easy":
            score_wait = c01(1.0 - avg_wait / cfg.score_refs["wait"])
            score_thr = c01(throughput / cfg.score_refs["thr"])
            score_rej = c01(1.0 - rejection_rate / cfg.score_refs["rej"])
            score_sla = c01(1.0 - sla_rate / cfg.score_refs["sla"])
            score = 0.4 * score_wait + 0.3 * score_thr + 0.15 * score_rej + 0.15 * score_sla
            details = {
                "score_wait": round(score_wait, 4),
                "score_throughput": round(score_thr, 4),
                "score_rejection": round(score_rej, 4),
                "score_sla": round(score_sla, 4),
            }
        elif cfg.task_id == "medium":
            p95_u = self._percentile(self._wait_samples_urgent, 0.95)
            p95_n = self._percentile(self._wait_samples_normal, 0.95)
            urgent_sla = self._safe_div(self._metrics["sla_breaches_urgent"], max(1.0, self._metrics["completed_urgent"]))
            s_uw = c01(1.0 - p95_u / cfg.score_refs["uw"])
            s_nw = c01(1.0 - p95_n / cfg.score_refs["nw"])
            s_usla = c01(1.0 - urgent_sla / cfg.score_refs["usla"])
            s_thr = c01(throughput / cfg.score_refs["thr"])
            s_cost = c01(1.0 - self._metrics["action_cost"] / cfg.score_refs["cost"])
            score = 0.35 * s_uw + 0.15 * s_nw + 0.25 * s_usla + 0.15 * s_thr + 0.10 * s_cost
            details = {
                "score_urgent_wait": round(s_uw, 4),
                "score_normal_wait": round(s_nw, 4),
                "score_urgent_sla": round(s_usla, 4),
                "score_throughput": round(s_thr, 4),
                "score_cost": round(s_cost, 4),
            }
        else:
            e2e_p95 = self._percentile(self._e2e_wait_samples, 0.95)
            abd_rate = self._safe_div(self._metrics["abandoned"], arrivals)
            s_e2e = c01(1.0 - e2e_p95 / cfg.score_refs["e2e"])
            s_abd = c01(1.0 - abd_rate / cfg.score_refs["abd"])
            s_sla = c01(1.0 - sla_rate / cfg.score_refs["sla"])
            s_thr = c01(throughput / cfg.score_refs["thr"])
            s_cost = c01(1.0 - self._metrics["infra_cost"] / cfg.score_refs["cost"])
            s_fair = c01(1.0 - fairness_gap / cfg.score_refs["fair"])
            score = 0.25 * s_e2e + 0.20 * s_abd + 0.20 * s_sla + 0.15 * s_thr + 0.10 * s_cost + 0.10 * s_fair
            details = {
                "score_e2e_p95": round(s_e2e, 4),
                "score_abandonment": round(s_abd, 4),
                "score_sla": round(s_sla, 4),
                "score_throughput": round(s_thr, 4),
                "score_cost": round(s_cost, 4),
                "score_fairness": round(s_fair, 4),
            }

        if self._metrics["invalid_actions"] > max(3.0, 0.04 * cfg.horizon):
            score = min(score, 0.4)
        # Apply strict open-interval clamp: validator rejects 0.0 and 1.0.
        return strict01(score), details

    def _build_observation(self, reward: float, done: bool, info: dict) -> CloudQueueObservation:
        cfg = self._task_configs[self._active_task_id]
        queue_lengths = [len(q) for q in self._queues]
        for i, q in enumerate(self._queues):
            current_mean_wait = 0.0
            if q:
                current_mean_wait = sum(job["wait"] for job in q) / len(q)
            self._wait_ema[i] = 0.8 * self._wait_ema[i] + 0.2 * current_mean_wait

        active_servers = max(1, sum(1 for s in self._servers if s["active"]))
        completed = max(1.0, self._metrics["completed"])
        sla_violation_rate = self._safe_div(self._metrics["sla_breaches"], completed)
        abandonment_rate = self._safe_div(self._metrics["abandoned"], max(1.0, self._metrics["arrivals"]))
        throughput_recent = max(0.0, info.get("completed_this_step", 0.0))
        energy_cost_rate = active_servers * cfg.server_cost

        incoming = self._incoming_job
        incoming_present = incoming is not None
        incoming_size = float(incoming["size"]) if incoming_present else 0.0
        incoming_priority = int(incoming["priority"]) if incoming_present else 0
        incoming_deadline = float(incoming["deadline"]) if incoming_present else 0.0
        incoming_type = int(incoming["type"]) if incoming_present else 0

        score, score_details = (0.0, {})
        if done:
            score, score_details = self._score_task(cfg)

        metadata = {
            "info": info,
            "reward_components": info.get("reward_components", {}),
            "applied_action": info.get("applied_action", "noop"),
            "seed": int(self._pending_seed),
            "trace_digest": self._trace_digest(),
            "rng_stream_seeds": self._rng_stream_seeds,
            "metrics": {
                "arrivals": self._metrics["arrivals"],
                "accepted": self._metrics["accepted"],
                "rejected": self._metrics["rejected"],
                "completed": self._metrics["completed"],
                "abandoned": self._metrics["abandoned"],
                "invalid_actions": self._metrics["invalid_actions"],
                "harmful_scale_down": self._metrics["harmful_scale_down"],
                "infra_cost": round(self._metrics["infra_cost"], 4),
            },
            "episode_score": round(score, 4),
            "score_details": score_details,
        }

        return CloudQueueObservation(
            task_id=cfg.task_id,
            sim_time=self._sim_time,
            horizon=cfg.horizon,
            queue_lengths=queue_lengths,
            queue_wait_ema=[round(v, 3) for v in self._wait_ema],
            server_busy=[1 if s["job"] is not None and s["active"] else 0 for s in self._servers],
            server_remaining_service=[round(float(s["remaining"]), 3) for s in self._servers],
            utilization=[round(v, 3) for v in self._utilization_ema[: len(self._servers)]],
            incoming_job_present=incoming_present,
            incoming_job_size=round(incoming_size, 3),
            incoming_job_priority=incoming_priority,
            incoming_job_deadline=round(incoming_deadline, 3),
            incoming_job_type=incoming_type,
            sla_violation_rate=round(sla_violation_rate, 4),
            abandonment_rate=round(abandonment_rate, 4),
            throughput_recent=round(throughput_recent, 4),
            energy_cost_rate=round(energy_cost_rate, 4),
            level=cfg.level,
            optional_history=[round(v, 4) for v in list(self._recent_rewards)],
            action_mask=[1 for _ in range(8)],
            done=done,
            reward=round(reward, 6),
            metadata=metadata,
        )

    def step(self, action: CloudQueueAction) -> CloudQueueObservation:  # type: ignore[override]
        cfg = self._task_configs[self._active_task_id]

        if (action.action_type or "").lower() == "configure_task":
            ok, note = self._apply_action(action, cfg)
            info = {
                "event": "configure_task",
                "applied_action": action.action_type,
                "valid_action": ok,
                "note": note,
                "completed_this_step": 0.0,
                "debug_trace_id": self._trace_digest(),
            }
            return self._build_observation(reward=0.0, done=self._done, info=info)

        if self._done:
            info = {
                "event": "episode_done",
                "applied_action": action.action_type,
                "valid_action": False,
                "note": "call reset() to start a new episode",
                "completed_this_step": 0.0,
                "reward_components": {},
                "debug_trace_id": self._trace_digest(),
            }
            return self._build_observation(reward=0.0, done=True, info=info)

        self._state.step_count += 1
        self._sim_time += 1

        completed_this_step = self._process_servers(cfg)
        abandoned_this_step = self._update_wait_and_abandonment(cfg)
        self._spawn_incoming_job(cfg)

        action_ok, action_note = self._apply_action(action, cfg)
        action_key = (
            f"{(action.action_type or 'noop').lower()}|"
            f"q={action.target_queue}|s={action.target_server}|"
            f"d={action.scale_delta}|p={action.new_priority}"
        )
        self._action_trace.append(action_key)
        self._autodispatch()
        reward, reward_components = self._compute_reward(
            cfg,
            action_ok=action_ok,
            action_type=(action.action_type or "noop").lower(),
            action_scale_delta=int(action.scale_delta or 0),
            completed_step=completed_this_step,
        )

        self._done = self._state.step_count >= cfg.horizon
        info = {
            "event": "step",
            "applied_action": action.action_type,
            "valid_action": action_ok,
            "note": action_note,
            "completed_this_step": completed_this_step,
            "abandoned_this_step": abandoned_this_step,
            "reward_components": reward_components,
            "debug_trace_id": self._trace_digest(),
        }
        return self._build_observation(reward=reward, done=self._done, info=info)

    @property
    def state(self) -> State:
        return self._state
