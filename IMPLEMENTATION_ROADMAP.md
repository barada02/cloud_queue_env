# QueueOps OpenEnv Implementation Roadmap

This file is the execution reference for building and iterating the queue operations environment.

Scope constraints:
- Keep current repository structure unchanged.
- Use cloud_queue_env as the project root.
- Follow OpenEnv compliance strictly.
- Provide deterministic graders with partial scores in [0, 1].
- Keep at least 3 benchmark tasks (easy, medium, hard).

---

## V1 - MVP Submission Build

Goal: ship a complete, valid benchmark that can be submitted.

### Phase 1 - Environment Core
Sub-goals:
1. Replace template echo behavior with queue simulator dynamics.
2. Implement deterministic state transitions using explicit seeds.
3. Implement terminal conditions with fixed task horizons.
4. Keep OpenEnv contract: reset, step, state.

Exit criteria:
1. reset/step/state are stable and deterministic for fixed seed + fixed action trace.
2. Episodes terminate correctly.

### Phase 2 - Task Pack (Easy/Medium/Hard)
Sub-goals:
1. Add task selector and fixed per-task configs.
2. Easy: single queue with admission/dispatch control.
3. Medium: multi-server with class-aware routing.
4. Hard: two-stage queue network with scaling decisions.

Exit criteria:
1. All three tasks run end-to-end.
2. Difficulty progression is visible from easy to hard.

### Phase 3 - Deterministic Graders
Sub-goals:
1. Implement per-task score equations with partial credit.
2. Clamp all task scores to [0, 1].
3. Handle edge cases (NaN/Inf/missing metrics) safely.
4. Add final aggregate score across tasks.

Exit criteria:
1. Same seeds and same actions always produce the same score.
2. Scores are interpretable and bounded.

### Phase 4 - Reward Shaping
Sub-goals:
1. Add dense multi-component rewards (wait, throughput, SLA, cost, fairness, safety).
2. Penalize invalid and exploit-like behavior.
3. Keep reward scale bounded and stable.
4. Expose component breakdown in metadata/info.

Exit criteria:
1. Reward changes across trajectory (not terminal-only).
2. Unsafe behavior is consistently penalized.

### Phase 5 - Inference Runner
Sub-goals:
1. Run all benchmark tasks with fixed seeds.
2. Use OpenAI-compatible client with provider credentials from env variables.
3. Emit [START], [STEP], [END] logs and final [SUMMARY].
4. Keep runs reproducible (fixed model params).

Exit criteria:
1. End-to-end benchmark run works locally and on deployed runtime.
2. Output format is submission-ready.

### Phase 6 - Validation and Docs
Sub-goals:
1. Pass openenv validate.
2. Ensure Docker build/run path works.
3. Update README with task, reward, grading, and baseline usage.
4. Add sample benchmark output snippet for evidence.

Exit criteria:
1. Validation passes.
2. README is complete for judges and users.

### V1 Submission Gate
All items must be true:
1. Three tasks implemented and deterministic.
2. Graders produce valid partial scores in [0, 1].
3. Inference script runs all tasks and reports summary.
4. OpenEnv validation passes.
5. Deployment path is functional.

---

## V2 - Robustness and Quality Upgrade

Goal: improve reliability, calibration, and benchmark trustworthiness.

### Phase 1 - Determinism Hardening
Sub-goals:
1. Separate RNG streams for arrivals/service/abandonment/shocks.
2. Add replay trace mode for debugging.
3. Add deterministic episode metadata for audits.

### Phase 2 - Difficulty Calibration
Sub-goals:
1. Tune easy/medium/hard parameter separation.
2. Improve anti-exploit balancing (reject-all, noop loops, over-scaling).
3. Re-check reward and grade alignment across seeds.

### Phase 3 - Reporting Upgrade
Sub-goals:
1. Add per-seed result table.
2. Add mean/std and confidence summary.
3. Add failure/invalid-action diagnostics in summary.

### V2 Exit Criteria
1. Lower variance for fixed seed sets.
2. Clearer task progression and fairer scoring.
3. Better debugging and reproducibility outputs.

---

## V3 - Extended Benchmark Pack

Goal: increase novelty and long-term benchmark value.

### Phase 1 - Optional Task D
Sub-goals:
1. Add stronger non-stationary demand patterns.
2. Grade robustness to bursts and demand shifts.

### Phase 2 - Optional Task E
Sub-goals:
1. Add partial observability/noisy delayed metrics.
2. Grade safe decision-making under uncertainty.

### Phase 3 - Public Benchmarking Bundle
Sub-goals:
1. Publish official seed suites and profiles (quick/standard/full).
2. Provide reference baseline runs.
3. Provide reproducibility notes for external users.

### V3 Exit Criteria
1. Four or more tasks available.
2. Stronger novelty and benchmark coverage.
3. Cleaner external benchmarking workflow.

---

## Recommended Execution Order

1. Complete V1 and submit.
2. Upgrade to V2 for reliability and scoring quality.
3. Add V3 only if timeline permits.

## Current Status Snapshot

1. V1 core implementation is in place and running.
2. openenv validate has passed.
3. V2 determinism hardening, calibration pass, and reporting upgrade are implemented.
4. Current focus shifts to V3 extensions and benchmark quality tuning.

## V2 Completion Notes

Implemented outcomes:
1. Separate RNG streams are active for arrivals, service, abandonment, and exogenous effects.
2. Deterministic trace metadata is exposed (`trace_digest`, `seed`, and RNG stream seeds).
3. Anti-exploit reward calibration includes rejection-heavy and harmful downscale penalties.
4. Inference supports multi-seed reporting with mean/std/ci95 outputs.
5. Inference supports replay-mode action traces via file input for deterministic debugging.
6. Inference supports JSON/CSV report export for per-seed analysis.

---

## Requirement Coverage Matrix (From requirementInfo.md)

This section is the final compliance tracker for judging criteria.

### Functional Requirements

1. Real-world task simulation
- Requirement: Must represent real human operational work, not toy behavior.
- Implementation target: queue operations in call center/cloud/logistics-style flow.
- Evidence to keep: README motivation + task descriptions + action semantics.
- Status: in progress (core done, examples and narrative should be strengthened).

2. OpenEnv spec compliance
- Requirement: typed models, reset, step(action), state, openenv.yaml, validate pass.
- Implementation target: models.py + server environment + openenv.yaml + app entrypoint.
- Evidence to keep: `openenv validate` output in PR notes/README.
- Status: done (validate passing).

3. Minimum 3 tasks with deterministic graders
- Requirement: at least easy/medium/hard, deterministic 0.0-1.0 grading.
- Implementation target: task configs + per-task scoring formulas + clamping.
- Evidence to keep: sample run showing all tasks and deterministic seeds.
- Status: done for 3 tasks, polish recommended for calibration.

4. Meaningful reward function
- Requirement: dense trajectory signal + penalties for undesirable behavior.
- Implementation target: weighted reward components and safety penalties.
- Evidence to keep: reward component logging in metadata and README equations.
- Status: done, tune weights in V2.

5. Baseline inference script
- Requirement: OpenAI-compatible client, env vars credentials, reproducible score over tasks.
- Implementation target: fixed tasks/seeds/model params, required log format.
- Evidence to keep: saved run logs and summary scores.
- Status: done, provider-fallback robustness can be improved.

### Non-Functional Requirements

1. Hugging Face Space deployment
- Requirement: containerized HF Space tagged openenv.
- Evidence to keep: Space URL + successful run proof.
- Status: done.

2. Containerized execution
- Requirement: Dockerfile works with build + run.
- Evidence to keep: commands and successful output snippet.
- Status: pending explicit evidence capture in docs.

3. Documentation completeness
- Requirement: README includes env motivation, spaces, tasks, setup/usage, baseline scores.
- Evidence to keep: README sections + benchmark output table.
- Status: mostly done, baseline score table still needed.

---

## Evaluation Criteria Coverage Checklist

### Real-world utility (30%)
1. Keep README examples tied to concrete real operations scenarios.
2. Add one paragraph on why this benchmark is useful for agent evaluation.

### Task and grader quality (25%)
1. Keep deterministic seed set fixed and documented.
2. Show per-task scoring decomposition and bounded outputs.
3. Add one reproducibility check note: same seed + same policy => same score.

### Environment design (20%)
1. Verify clean reset and sensible done boundaries for all tasks.
2. Keep action/observation schema stable and documented.
3. Keep dense reward with interpretable components.

### Code quality and spec compliance (15%)
1. Keep `openenv validate` passing.
2. Capture docker build/run commands and outcomes.
3. Keep deployment and ws route functional.

### Creativity and novelty (10%)
1. Emphasize queue-control benchmark novelty in README.
2. Keep multi-objective reward and cost/fairness tradeoff visible.

---

## Pre-Submission Evidence Pack (Must Attach)

1. Validation proof
- `openenv validate` success output.

2. Runtime proof
- HF Space URL and one successful task run excerpt.

3. Baseline proof
- One full [START]/[STEP]/[END]/[SUMMARY] run log.

4. Docker proof
- `docker build` and `docker run` command results.

5. Documentation proof
- README includes baseline score table (easy, medium, hard, final).
