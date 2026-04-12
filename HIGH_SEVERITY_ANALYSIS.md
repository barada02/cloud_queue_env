# Cloud Queue Env - High Severity Analysis (Updated)

Date: 2026-04-12

This note captures the two highest-impact issues still present in the environment logic.

## 1) Arrival Modeling and Arrival Metrics Mismatch

Files and lines:
- cloud_queue_env/server/cloud_queue_env_environment.py:240
- cloud_queue_env/server/cloud_queue_env_environment.py:241
- cloud_queue_env/server/cloud_queue_env_environment.py:248
- cloud_queue_env/server/cloud_queue_env_environment.py:259

What happens now:
- The simulator samples Poisson arrivals each step.
- If sampled arrivals are greater than 1, the code still creates only one incoming job object.
- The arrivals metric is incremented by 1.0, not by sampled arrival count.

Why this is high severity:
- Burst behavior is compressed into a single-event stream, so load spikes are underrepresented.
- Several business metrics and grader components become biased (rejections, abandonment, SLA pressure).
- Policy ranking can drift because the environment under-penalizes burst scenarios.

Impact on benchmark credibility:
- High. This directly affects realism, fairness of grading, and reproducibility quality claims.

Recommended fix direction:
- Track all sampled arrivals each step.
- Either queue all arrivals or maintain an explicit backlog of pending incoming jobs.
- Increment arrivals metric using true sampled count.

## 2) Agent Dispatch Control Is Partially Bypassed by Autodispatch

Files and lines:
- cloud_queue_env/server/cloud_queue_env_environment.py:353
- cloud_queue_env/server/cloud_queue_env_environment.py:391
- cloud_queue_env/server/cloud_queue_env_environment.py:738

What happens now:
- The agent may choose an action that is not dispatch.
- After action application, the environment still runs autodispatch and moves work to idle servers.

Why this is high severity:
- It weakens action-to-outcome causality for dispatch decisions.
- A policy can look better than it should because server assignment still happens automatically.
- It reduces benchmark difficulty in exactly the control surface the task is evaluating.

Impact on benchmark credibility:
- High. This can alter policy comparisons and invalidate assumptions about explicit control.

Recommended fix direction:
- Make dispatch behavior explicit by mode:
  - strict-control mode: only agent dispatches.
  - assisted mode: autodispatch on, but document this clearly and score accordingly.
- Keep one consistent mode for official benchmark scoring.

## Priority Summary

1. Fix arrival accounting and multi-arrival handling first.
2. Fix dispatch authority semantics second.

Both should be addressed before claiming benchmark-grade reliability.
