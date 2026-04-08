# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Cloud Queue Env queue operations environment."""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CloudQueueAction(Action):
    """Action model for queue control decisions."""

    action_type: str = Field(
        default="noop",
        description=(
            "One of: configure_task, admit, reject, route, dispatch, scale, reprioritize, noop"
        ),
    )
    target_queue: int | None = Field(default=None, description="Queue index for route/dispatch")
    target_server: int | None = Field(default=None, description="Server index for dispatch")
    scale_delta: int | None = Field(default=None, description="Server pool scale delta for scale action")
    new_priority: int | None = Field(default=None, description="Updated priority for reprioritize action")
    task_id: str | None = Field(default=None, description="Task selector: easy, medium, or hard")
    seed: int | None = Field(default=None, description="Deterministic seed for upcoming reset")


class CloudQueueObservation(Observation):
    """Observation model exposing queue system state to the agent."""

    task_id: str = Field(default="easy", description="Active benchmark task")
    sim_time: int = Field(default=0, description="Discrete simulation time step")
    horizon: int = Field(default=0, description="Episode horizon")
    queue_lengths: list[int] = Field(default_factory=list, description="Length per queue")
    queue_wait_ema: list[float] = Field(default_factory=list, description="EMA wait time per queue")
    server_busy: list[int] = Field(default_factory=list, description="1 if server is busy, else 0")
    server_remaining_service: list[float] = Field(
        default_factory=list,
        description="Remaining service time per server",
    )
    utilization: list[float] = Field(default_factory=list, description="Rolling utilization by server")
    incoming_job_present: bool = Field(default=False, description="Whether a new job is waiting for admission")
    incoming_job_size: float = Field(default=0.0, description="Incoming job estimated size")
    incoming_job_priority: int = Field(default=0, description="Incoming job priority")
    incoming_job_deadline: float = Field(default=0.0, description="Incoming job deadline")
    incoming_job_type: int = Field(default=0, description="Incoming job class/type id")
    sla_violation_rate: float = Field(default=0.0, description="Running SLA violation rate")
    abandonment_rate: float = Field(default=0.0, description="Running abandonment rate")
    throughput_recent: float = Field(default=0.0, description="Completed jobs in current step")
    energy_cost_rate: float = Field(default=0.0, description="Current infrastructure cost rate")
    level: float = Field(default=1.0, description="Difficulty level scalar")
    optional_history: list[float] = Field(default_factory=list, description="Compact recent context")
    action_mask: list[int] = Field(default_factory=list, description="Optional valid action hints")
