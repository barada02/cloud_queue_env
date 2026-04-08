# QueueOps OpenEnv Implementation Roadmap

This roadmap is the execution reference for building the real-world queueing environment in this repository.

Constraints locked in:
- Keep existing directory structure unchanged.
- Treat `cloud_queue_env/` as the project root.
- Use HF token provider flow in `inference.py`.
- Follow OpenEnv compliance strictly: typed models, `step()/reset()/state()`, valid `openenv.yaml`.
- Provide deterministic graders with partial scoring in `[0, 1]`.
- Deliver at least 3 tasks (more optional).

---

## V1 - Hackathon-Ready Submission

Goal: submit a valid, real-world OpenEnv benchmark with 3 deterministic graded tasks and reproducible inference outputs.

### Phase 1 - Core Simulator Foundation
Sub-goals:
1. Replace echo logic with queue-operations simulation core.
2. Add deterministic RNG with explicit seed handling.
3. Implement proper episode boundaries (`horizon`, terminal conditions).
4. Keep strict OpenEnv contract for `reset()`, `step()`, and `state`.

Definition of done:
- Environment no longer behaves as dummy echo.
- Same seed + same action trace => identical trajectory.
- Episode always terminates predictably.

### Phase 2 - Task System (Easy/Medium/Hard)
Sub-goals:
1. Add task selection (`task_id`) and per-task config.
2. Implement Task A (single queue, admission control).
3. Implement Task B (multi-server, priority routing).
4. Implement Task C (two-stage queue network, dynamic scaling/cost).

Definition of done:
- All 3 tasks run end-to-end from `reset()` to terminal state.
- Difficulty progression is visible from A -> B -> C.

### Phase 3 - Deterministic Graders + Partial Scoring
Sub-goals:
1. Implement per-task grader formulas from master spec.
2. Keep each grader output bounded in `[0, 1]`.
3. Handle invalid/NaN/infinite values safely and deterministically.
4. Aggregate final benchmark score as mean of task scores.

Definition of done:
- Repeated runs on same seeds produce same grader outputs.
- Partial scoring is meaningful (not binary pass/fail only).

### Phase 4 - Reward Shaping and Safety Penalties
Sub-goals:
1. Add dense reward components: wait, throughput, SLA, cost, fairness, safety.
2. Add penalties for invalid actions and exploit patterns.
3. Bound reward scale across tasks.
4. Expose reward components in `info` for debugging.

Definition of done:
- Reward moves through trajectory, not only at the end.
- Unsafe or degenerate behavior is penalized.

### Phase 5 - Inference Protocol Compliance
Sub-goals:
1. Update `inference.py` to run all required tasks with fixed seeds.
2. Keep OpenAI client usage while authenticating with HF token flow.
3. Emit strict `[START]`, `[STEP]`, `[END]` line format.
4. Print per-task and final aggregate scores.

Definition of done:
- Script executes benchmark sweep reproducibly.
- Output format matches hackathon requirements.

### Phase 6 - Packaging, Validation, Documentation
Sub-goals:
1. Validate `openenv.yaml` metadata and app wiring.
2. Confirm Docker build/run success.
3. Update README with task definitions, action/observation spaces, reward/grader equations, baseline results.
4. Verify deployment readiness for HF Space.

Definition of done:
- OpenEnv validation passes.
- Container starts and serves correctly.
- README is submission-ready.

### V1 Submission Gate
All must be true:
1. 3 tasks implemented and deterministic.
2. Graders return valid partial scores in `[0, 1]`.
3. Inference script reports reproducible benchmark outputs.
4. OpenEnv spec compliance confirmed.
5. Docker and README requirements satisfied.

---

## V2 - Quality and Robustness Upgrade

Goal: improve benchmark reliability, score stability, and anti-exploit behavior after initial submission.

### Phase 1 - Determinism Hardening
Sub-goals:
1. Split RNG streams (arrivals/service/abandonment/shocks).
2. Add trace replay support for debugging.
3. Extend `info` with deterministic audit fields.

### Phase 2 - Difficulty Calibration
Sub-goals:
1. Tune parameters for cleaner A/B/C separation.
2. Improve level interpolation behavior.
3. Add stronger guards against reject-all or noop exploitation.

### Phase 3 - Reporting and Confidence
Sub-goals:
1. Add standardized per-seed report table.
2. Add mean/std summaries over seed sets.
3. Flag unstable metrics and grader edge cases.

### V2 Exit Criteria
1. Lower run-to-run variance on fixed seed sets.
2. Clearer task difficulty progression.
3. Better fairness and exploit resistance.

---

## V3 - Extended Benchmark Pack (Optional)

Goal: increase novelty and long-term benchmark value with optional extra tasks.

### Phase 1 - Task D (Non-stationary Load)
Sub-goals:
1. Add shift-based and bursty arrivals.
2. Grade robustness under changing demand.

### Phase 2 - Task E (Partial Observability)
Sub-goals:
1. Add delayed/noisy metrics.
2. Grade safe decisions under uncertainty.

### Phase 3 - Public Benchmark Packaging
Sub-goals:
1. Publish official seed suites.
2. Add benchmark profiles: quick / standard / full.
3. Provide reference baseline outputs.

### V3 Exit Criteria
1. 4-5 total tasks available.
2. Broader real-world coverage.
3. Stronger benchmark differentiation.

---

## Execution Order

Recommended order:
1. Complete V1 fully and submit.
2. Continue with V2 for quality hardening.
3. Do V3 only if timeline allows.

Immediate next implementation step:
- Start V1 Phase 1 (models + simulator core + deterministic state transitions).
