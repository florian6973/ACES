# Compiled Polars Engine (design)

> Status: **design / in progress**. This document specifies a new, opt-in execution
> backend for ACES that *compiles* a task configuration into a single fused Polars
> `LazyFrame` query plan, as an alternative to the recursive interpreter in
> [`extract_subtree.py`](../../src/aces/extract_subtree.py).

## 1. Motivation

ACES is already Polars-based, but `query()` runs a **recursive Python interpreter**
over the window tree ([`extract_subtree.extract_subtree`](../../src/aces/extract_subtree.py)).
At every tree node it:

1. aggregates predicates over the **entire** `predicates_df` *eagerly*
   (`aggregate_temporal_window` / `aggregate_event_bound_window`),
2. materializes an intermediate `DataFrame`,
3. inner-joins it to the current anchor set,
4. filters by constraints, and
5. recurses into each child.

This materialize-at-every-node strategy is the root of the cost reported in
[`profiling.md`](profiling.md): on a single MIMIC-IV shard (~50k subjects, ~80M
events) tasks take **~180–370 s and ~30–106 GB** of memory.

### Key insight

The window tree is **statically known at compile time** — the control flow does not
depend on the data, only the *values* do. Moreover each anchor realizes **at most one**
child anchor:

- a *temporal* window keeps the same anchor timestamp;
- an *event-bound* window (`boolean_expr_bound_sum`) emits exactly one row per input
  row, i.e. one boundary timestamp per anchor.

So the recursion **never fans out combinatorially**. That means it can be **unrolled at
compile time into one fused `LazyFrame`** and handed to the Polars streaming engine,
which is what unlocks the time and (especially) memory improvements.

## 2. Goals & non-goals

**Goals**

- Full feature parity with the existing engine (temporal *and* event-bound windows,
  derived predicates, constraints, static variables, `label`, `index_timestamp`,
  multi-branch trees, `_RECORD_START`/`_RECORD_END`, offsets, negative windows).
- Opt-in: selectable via `engine={legacy,compiled}`; **legacy remains the default**
  until parity is proven.
- Bit-for-bit equal output to `query()` (after canonical sort), enforced by a
  differential test harness over all `sample_configs/` and existing fixtures.
- A reproducible synthetic benchmark quantifying time + peak memory vs. the legacy
  engine across dataset sizes.

**Non-goals (initial)**

- Changing the YAML schema or the `TaskExtractorConfig` parser.
- Changing predicate generation (`predicates.py`) or I/O (`run.py`).
- Distributed / multi-shard execution (orthogonal; handled upstream by sharding).

## 3. Architecture

```
TaskExtractorConfig  ──compile_query()──▶  (pl.LazyFrame) -> pl.LazyFrame   [pure plan]
predicates_df / LazyFrame ─────────────────────────────────────────────────┐
                                                                            ▼
                                              lazy_query(cfg, preds) -> pl.LazyFrame
                                                                            │
                                                          .collect(engine="streaming")
                                                          (eager fallback if needed)
```

**New modules**

| Module | Responsibility |
| --- | --- |
| `src/aces/compile.py` | `compile_query(cfg)` — walk `cfg.window_tree` preorder, emit lazy ops per node, return a plan-builder. |
| `src/aces/lazy_query.py` | `lazy_query(cfg, predicates)` — public entry mirroring `query()`'s signature and output columns. |
| `src/aces/_window_exprs.py` | Shared **single source of truth** for temporal + event-bound window semantics, used by *both* engines to prevent parity drift. |

**Wiring**

- `run.py` / Hydra config gain `engine: legacy | compiled` (default `legacy`).
- `query()` is untouched; `lazy_query().collect()` is the compiled path.

## 4. Compilation strategy

Let `base = lf.sort("subject_id", "timestamp")` (the predicates frame as a `LazyFrame`).

The plan is built by unrolling `extract_subtree` over the static tree. Each window node
`child` of parent `p` becomes:

1. **Window summary expression** keyed by `(subject_id, anchor_ts)`:
   - *Temporal* (`endpoint_expr[1]` is a `timedelta`): a rolling aggregation built from
     the existing `TemporalWindowBounds.polars_gp_rolling_kwargs`
     ([`types.py`](../../src/aces/types.py)), expressed lazily. Anchor = row timestamp.
   - *Event-bound* (`endpoint_expr[1]` is a `str`): a **lazy reimplementation** of
     `boolean_expr_bound_sum` ([`aggregate.py`](../../src/aces/aggregate.py)) using
     `cum_sum().over("subject_id")` plus forward/back-fill of the boundary timestamp.
     Anchor = realized boundary timestamp (`timestamp_at_start` for `-`-prefixed /
     `bound_to_row`, else `timestamp_at_end`).
   - The accumulated `subtree_root_offset` is folded into `endpoint_expr.offset` at
     compile time (exactly as the interpreter does at
     [`extract_subtree.py:292`](../../src/aces/extract_subtree.py#L292)).
2. **Join** the summary onto the current anchor frame on `(subject_id, anchor_ts)`.
3. **Constraint filter**: translate `child.constraints` (`{pred: (min, max)}`) into a
   `.filter()` predicate, matching `check_constraints` semantics exactly (including the
   `*` / `_ANY_EVENT` wildcard).
4. **Child anchor** = the realized boundary timestamp column; recurse.
5. **Branches**: multiple children inner-join on the shared root anchor
   `(subject_id, subtree_anchor_timestamp)` — mirroring step 7 of the interpreter.

Finally, assemble `_summary` structs per node, `trigger`, `label`, and
`index_timestamp` exactly as [`query.py:153-197`](../../src/aces/query.py#L153),
and select the same output column order.

### Why this is faster / leaner

- **One optimized graph**: predicate pushdown, projection pushdown, and common
  sub-expression handling run across the whole task, not per node.
- **Streaming execution**: bounded memory instead of materializing a full
  per-row aggregation of the entire frame at every recursion step (the source of the
  106 GB peaks).
- **No Python round-trips**: the per-node `DataFrame` → join → filter loop collapses
  into native Polars.

### Highest-risk component

The lazy reimplementation of `boolean_expr_bound_sum` — the
`mode ∈ {row_to_bound, bound_to_row} × closed ∈ {both,left,right,none} × offset`
matrix is subtle (see the truth table in the function's docstring). This is covered by
dedicated parity tests (§5) and is the first thing to land and verify.

## 5. Correctness & testing

Parity is enforced by a **differential oracle**: the legacy engine *is* the spec.

- `tests/test_compiled_parity.py`: for every `sample_configs/*.yaml` and every existing
  fixture (`test_e2e`, `test_meds`, `test_other_meds`, `test_override_meds`), run both
  `query(cfg, df)` and `lazy_query(cfg, df).collect()` and `assert_frame_equal` after a
  canonical sort of rows and columns.
- Targeted per-feature parity tests where the compile is subtle: each `closed` mode,
  `row_to_bound`/`bound_to_row`, positive/negative/zero offsets, negative windows,
  `_RECORD_START`/`_RECORD_END`, static predicates, empty results, multi-branch trees,
  and `label`/`index_timestamp` selection.
- Extend [`test_aggregate_hypothesis.py`](../../tests/test_aggregate_hypothesis.py):
  feed Hypothesis-generated frames through both the legacy and lazy window builders and
  assert equality (fuzzes the `boolean_expr_bound_sum` reimplementation).
- Reuse
  [`test_extract_subtree_idempotency.py`](../../tests/test_extract_subtree_idempotency.py)
  scenarios against the compiled path.

A streaming-unsupported op triggers an automatic fallback to in-memory `.collect()`;
both paths are asserted equal in tests.

## 6. Benchmark plan (synthetic)

- `benchmarks/generate.py`: deterministic, seeded parametric generator producing a
  MEDS-style predicates parquet. Parameters: `n_subjects`, events-per-subject
  distribution, number of predicates, predicate sparsity. (No wall-clock/random-seed
  hazards — seed is an explicit arg.)
- `benchmarks/bench.py`: sweep `n_subjects` (e.g. 1k → 1M), run both engines on each
  `sample_configs/` task, record:
  - wall time via `time.perf_counter`,
  - peak memory via `psutil` peak working set + `tracemalloc` for Python allocations,
  - assert output equality at small sizes (the benchmark double-checks parity).
- Outputs: `benchmarks/results/*.csv` plus matplotlib plots (time-vs-N and
  peak-mem-vs-N, legacy vs compiled), committed for reference alongside
  [`profiling.md`](profiling.md).

## 7. Phased delivery

1. **Scaffolding + harness**: branch, module stubs, synthetic generator, differential
   test harness wired over `sample_configs/` (initially `xfail`).
2. **Temporal-only compiler**: temporal windows, constraints, static vars, label/index,
   multi-branch. Differential tests green for temporal-only configs.
3. **Event-bound compiler**: lazy `boolean_expr_bound_sum`; full parity across all
   `sample_configs/` and fixtures.
4. **Engine flag**: `engine=compiled` in `run.py`/Hydra + `lazy_query` public API.
5. **Benchmark**: run the sweep, commit results + plots, write up findings here.

## 8. Open edge cases tracked during implementation

- `(subject_id, timestamp)` uniqueness assumption (already enforced by `query()`).
- Row-order of output: differential comparison sorts first; document that the compiled
  engine does not promise the legacy row order unless we add an explicit final sort.
- Confirm no anchor fan-out for event-bound windows under all `closed` modes.
- Null-timestamp / static-variable handling parity (`check_static_variables`).
- Empty-result short-circuits returning `pl.DataFrame()` with matching schema.
