# Compiled Polars Engine

> Status: **implemented and opt-in, but NOT recommended for performance.** This document
> specifies a new execution backend for ACES that *compiles* a task configuration into a
> single fused Polars `LazyFrame` plan, as an alternative to the recursive interpreter in
> [`extract_subtree.py`](../../src/aces/extract_subtree.py). It is fully correct (bit-for-bit
> parity with the legacy engine, enforced by tests), but benchmarking showed it is **slower
> and uses more memory** than the legacy engine at every size tested. See
> [§6 Findings](#6-findings-measured). It ships behind `engine=compiled` for reproducibility
> and as a foundation for the future, genuinely-optimized directions in [§7](#7-where-a-real-speedup-would-come-from).

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
compile time into one fused `LazyFrame`** and handed to the Polars streaming engine.

The *hypothesis* was that fusing the plan and streaming it would cut both time and peak
memory. The implementation below is a faithful, correct realization of that idea — and the
benchmark in [§6](#6-findings-measured) shows the hypothesis **does not hold**: a faithful
compile does the same work as the interpreter, and fusing the whole tree into one plan
*removes* the per-node materialize-and-free checkpoints that keep the interpreter's peak
memory low. The negative result is documented here because it is itself the useful finding.

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

### Highest-risk component

The lazy reimplementation of `boolean_expr_bound_sum` — the
`mode ∈ {row_to_bound, bound_to_row} × closed ∈ {both,left,right,none} × offset`
matrix is subtle (see the truth table in the function's docstring). This is covered by
dedicated parity tests (§5) and is the first thing to land and verify.

## 5. Correctness & testing

Parity is enforced by a **differential oracle**: the legacy engine *is* the spec.

- [`tests/test_window_exprs.py`](../../tests/test_window_exprs.py): pins the lazy window
  builders directly to `aggregate_temporal_window` / `aggregate_event_bound_window` across
  the full `mode × closed × offset × end_event` matrix on fixed and randomized frames
  (216 cases). This is where the highest-risk `boolean_expr_bound_sum` port is verified.
- [`tests/test_compiled_parity.py`](../../tests/test_compiled_parity.py): for every
  `sample_configs/*.yaml` and multiple seeds, runs both `query(cfg, df)` and
  `lazy_query(cfg, df)` and `assert_frame_equal` after a canonical sort.

A streaming-unsupported op triggers an automatic fallback to in-memory `.collect()` in
[`lazy_query`](../../src/aces/lazy_query.py).

## 6. Findings (measured)

Synthetic sweep (seed 0, 50 events/subject) on this machine (Windows, Polars 1.40), each
engine run in an **isolated subprocess** so peak working set is attributable to that engine
alone. Raw data: [`benchmarks/results/isolated.csv`](../../benchmarks/results) and the
`*_seconds.png` / `*_peak_wset_mb.png` plots beside it.

`inhospital_mortality` (event-bound-heavy: `-_RECORD_START`, `-> discharge_or_death`):

| subjects | pred rows | engine | time (s) | peak mem (MB) |
| --- | --- | --- | --- | --- |
| 80,000 | 4.08 M | legacy | **2.06** | **2,788** |
| 80,000 | 4.08 M | compiled (streaming) | 3.71 | 4,115 |
| 80,000 | 4.08 M | compiled (in-memory) | 2.75 | 3,352 |
| 200,000 | 10.2 M | legacy | **8.15** | **6,615** |
| 200,000 | 10.2 M | compiled (streaming) | 12.23 | 9,706 |
| 200,000 | 10.2 M | compiled (in-memory) | 8.66 | 8,590 |

`readmission_risk` (lighter, larger output):

| subjects | pred rows | engine | time (s) | peak mem (MB) |
| --- | --- | --- | --- | --- |
| 200,000 | 10.2 M | legacy | **3.11** | **4,123** |
| 200,000 | 10.2 M | compiled (streaming) | 9.47 | 4,406 |
| 200,000 | 10.2 M | compiled (in-memory) | 8.12 | 5,580 |

**The legacy engine wins on both time and memory at every size tested.** The compiled
streaming engine is consistently the slowest and the most memory-hungry. Why:

1. **No work is saved.** The compiled plan is a faithful translation of the recursion, so
   it performs the same aggregations and joins — there is no algorithmic reduction for the
   query optimizer to exploit, and the per-window summary contract forces every predicate
   column through every window (so projection pushdown can't prune them).
2. **Fusing removes free memory checkpoints.** The interpreter materializes one node's
   `DataFrame`, uses it, and lets Python free it before the next node. Collapsing the whole
   tree into one plan keeps far more intermediate state live simultaneously, *raising* peak
   memory rather than lowering it.
3. **Streaming overhead without streaming benefit.** Nothing spills at ≤10 M rows, so the
   streaming engine's morsel/pipeline machinery is pure overhead on this self-join- and
   struct-heavy plan.

The 106 GB peaks in [`profiling.md`](profiling.md) therefore are **not** explained by
"materializing at every node" — that materialization is actually load-bearing for keeping
memory bounded.

## 7. Where a real speedup would come from

A faithful compile cannot beat the interpreter; it does the same work with less freedom to
release memory. A genuinely faster engine would need *algorithmic* changes, not just lazy
translation:

- **Per-window column projection.** Only compute the predicate counts a window's
  constraints/label actually reference, instead of all predicates for all windows (requires
  changing the "every window summarizes every predicate" output contract, or computing the
  full summary lazily only for surviving rows).
- **Avoid full-frame re-aggregation per node.** Both engines aggregate over the entire frame
  at each node; restricting aggregation to the neighborhoods of surviving anchors before
  summarizing would cut work super-linearly on selective tasks.
- **Early subject pruning.** Drop subjects that cannot satisfy a constraint before
  descending, shrinking every downstream aggregation.
- **Chunked/streamed-by-subject execution** for the genuine out-of-core regime (the only
  place streaming should help), measured against real MIMIC-scale data rather than synthetic
  ≤10 M-row frames.

These are out of scope for the faithful-parity engine landed here, which exists to (a)
establish the differential oracle and benchmark harness and (b) make the negative result
reproducible.

## 8. Edge cases handled (parity-verified)

- `(subject_id, timestamp)` uniqueness assumption (already enforced by `query()`).
- Row-order of output: differential comparison sorts first; document that the compiled
  engine does not promise the legacy row order unless we add an explicit final sort.
- Confirm no anchor fan-out for event-bound windows under all `closed` modes.
- Null-timestamp / static-variable handling parity (`check_static_variables`).
- Empty-result short-circuits returning `pl.DataFrame()` with matching schema.
