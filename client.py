# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cloud Queue Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CloudQueueAction, CloudQueueObservation


class CloudQueueEnv(
    EnvClient[CloudQueueAction, CloudQueueObservation, State]
):
    """
    Client for the Cloud Queue Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with CloudQueueEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.queue_lengths)
        ...
        ...     result = client.step(CloudQueueAction(action_type="admit", target_queue=0))
        ...     print(result.observation.throughput_recent)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CloudQueueEnv.from_docker_image("cloud_queue_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CloudQueueAction(action_type="dispatch", target_queue=0))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CloudQueueAction) -> Dict:
        """
        Convert CloudQueueAction to JSON payload for step message.

        Args:
            action: CloudQueueAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type,
            "target_queue": action.target_queue,
            "target_server": action.target_server,
            "scale_delta": action.scale_delta,
            "new_priority": action.new_priority,
            "task_id": action.task_id,
            "seed": action.seed,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CloudQueueObservation]:
        """
        Parse server response into StepResult[CloudQueueObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CloudQueueObservation
        """
        obs_data = payload.get("observation", {})
        observation = CloudQueueObservation(
            task_id=obs_data.get("task_id", "easy"),
            sim_time=obs_data.get("sim_time", 0),
            horizon=obs_data.get("horizon", 0),
            queue_lengths=obs_data.get("queue_lengths", []),
            queue_wait_ema=obs_data.get("queue_wait_ema", []),
            server_busy=obs_data.get("server_busy", []),
            server_remaining_service=obs_data.get("server_remaining_service", []),
            utilization=obs_data.get("utilization", []),
            incoming_job_present=obs_data.get("incoming_job_present", False),
            incoming_job_size=obs_data.get("incoming_job_size", 0.0),
            incoming_job_priority=obs_data.get("incoming_job_priority", 0),
            incoming_job_deadline=obs_data.get("incoming_job_deadline", 0.0),
            incoming_job_type=obs_data.get("incoming_job_type", 0),
            sla_violation_rate=obs_data.get("sla_violation_rate", 0.0),
            abandonment_rate=obs_data.get("abandonment_rate", 0.0),
            throughput_recent=obs_data.get("throughput_recent", 0.0),
            energy_cost_rate=obs_data.get("energy_cost_rate", 0.0),
            level=obs_data.get("level", 1.0),
            optional_history=obs_data.get("optional_history", []),
            action_mask=obs_data.get("action_mask", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
