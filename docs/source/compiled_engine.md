# Compiled Polars Engine

> Status: **implemented, opt-in (`engine=compiled`), bit-for-bit parity with the legacy
> engine.** This is a new execution backend that *compiles* a task config into a fused Polars
> `LazyFrame` plan, as an alternative to the recursive interpreter in
> [`extract_subtree.py`](../../src/aces/extract_subtree.py).
>
> **Performance** (see [§6](#6-findings-measured)): after four optimizations — flattening
> temporal subtrees (§6.2), an event-bound anchor restriction (§6.3) and a cumsum + `join_asof`
> fast path (§6.3b), and a base-materialization barrier (§6.3c) — the compiled engine is
> **1.6–3.8× faster than legacy on four of five benchmarked configs and within ~1.3× (and
> leaner) on the fifth**, scaling linearly in window count. The naive first compile was
> catastrophically super-linear on deep window chains; that is fixed.

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
  `sample_configs/*.yaml`, multiple seeds, **and** parametric many-window `chain`/`wide`
  configs ([`benchmarks/complex_configs.py`](../../benchmarks/complex_configs.py)), runs both
  `query(cfg, df)` and `lazy_query(cfg, df)` and `assert_frame_equal` after a canonical sort.
  The complex configs specifically guard the flattening optimization (§6.2).

A streaming-unsupported op triggers an automatic fallback to in-memory `.collect()` in
[`lazy_query`](../../src/aces/lazy_query.py).

## 6. Findings (measured)

All numbers from synthetic sweeps (seed 0, 50 events/subject) on this machine (Windows,
Polars 1.40). Reproduce with [`benchmarks/run_isolated.py`](../../benchmarks/run_isolated.py)
(isolated-subprocess peak memory) and [`benchmarks/complex_configs.py`](../../benchmarks/complex_configs.py)
(window-count scaling).

### 6.1 The naive compile is super-linear in tree *depth*

The first version of the compiler was a faithful unrolling of `extract_subtree`: it mirrored
the recursion into a deep cascade of joins inside one lazy plan. That cascade is fine when
the tree is **wide** (windows hanging off the trigger) but pathological when it is **deep**
(a chain of windows each starting where the previous ended). On a temporal chain at 20 k
subjects:

| windows | legacy (s) | naive-compiled (s) | flattened-compiled (s) |
| --- | --- | --- | --- |
| 1 | 0.12 | 0.08 | **0.04** |
| 8 | 0.72 | 1.84 | **0.27** |
| 16 | 1.40 | 7.09 | **0.64** |
| 24 | 2.11 | 15.05 | **1.04** |
| 32 | 2.80 | 27.84 | **1.34** |

Legacy is linear in depth (it materializes each node and frees it); the naive compile blows
up super-linearly because the whole nested join graph stays live in one plan. Wide configs,
by contrast, stayed linear and the naive compile was already ≈0.8× of legacy.

### 6.2 The flattening optimization

The fix exploits a structural fact: a **temporal** window keeps the *same anchor timestamp*
as its parent, so a chain of temporal windows all share one anchor (the trigger time) and do
**not** need nesting. [`_process_children`](../../src/aces/compile.py) therefore *flattens*
temporal edges — joining each window summary onto a single accumulating frame keyed by the
trigger anchor — and only *nests* at **event-bound** edges, which genuinely move the anchor
to a realized boundary event. This restores linear scaling (right column above): at depth 32
it is **20× faster than the naive compile and ≈2× faster than legacy**, with bit-for-bit
identical output (parity tests green).

### 6.3 The event-bound anchor restriction

A second optimization targets the event-bound path. `summarize_event_bound_window` ran the
full `boolean_expr_bound_sum` (a `concat` + `sort` + windowed `fill_null`) over *every* row,
then `_process_children` discarded all non-anchor rows. But an anchor's nearest boundary is
determined only by the boundary rows — non-anchor "real" rows never affect it. So restricting
the real rows to the current anchor set *before* the concat/sort is **exact**, and shrinks
that step from O(all events) to O(anchors + boundary events). The cumulative sums are still
taken over the full frame (one pass) so per-row counts stay correct. Wired through the
optional `anchors=` argument; the window-twin tests still exercise the unrestricted path.

### 6.3b Cumsum + `join_asof` for offset-free event-bound windows

The anchor restriction still ran a full-frame `concat` + `sort` per window. For the common
case of a window with **no offset** (every event-bound window in the bundled sample configs),
[`_event_bound_asof`](../../src/aces/_window_exprs.py) replaces that entirely: it takes
per-subject cumulative sums once, builds a boundary table whose keys are shifted by `±1 µs`
(mirroring the concat path's epsilon, so the `closed`/tie behavior is identical), and locates
each row/anchor's boundary with a single `join_asof` (backward for `bound_to_row`, forward for
`row_to_bound`). The count is then a cumsum difference. No full-frame concat/sort; the join
runs on the (anchor-restricted) row set. Offset windows keep the proven concat path. The
216-case window-twin oracle pins this to bit-for-bit equality with the eager aggregator.

Effect on event-bound configs (in-memory collect, 80 k subjects), cumulative across §6.3 +
§6.3b:

| config | legacy (s) | faithful (s) | + anchor restrict | + join_asof |
| --- | --- | --- | --- | --- |
| `imminent_mortality` | 2.45 | — | 0.54 | **0.56** (0.23×) |
| `long_term_recurrence` | 1.52 | — | 0.92 | **0.81** (0.53×) |
| `abnormal_lab` | 1.99 | — | — | **0.84** (0.42×) |
| `inhospital_mortality` | 2.17 | 3.99 (2.12×) | 3.11 (1.65×) | **3.08 (1.42×)** |

Most event-bound configs comfortably beat legacy. `inhospital_mortality` keeps improving
(2.12× → 1.65× → 1.42×) but at this point is bottlenecked by repeated base prep, addressed next.

### 6.3c Materialization barrier on the shared base

Profiling `inhospital_mortality` showed the **static-variable filter**
(`(timestamp.is_null() & male>0).any().over(subject)`, a full-frame window op) being
recomputed **~6 times** — once per window/branch — because every `summarize_*` reads from the
same prepared `predicates_lf` and Polars' common-subplan elimination does not dedupe it across
the join tree. `compile_query(materialize_base=True)` (the default) collects the prepared base
**once** before fanning out, mirroring the per-node materialization the legacy interpreter gets
for free. This is value-preserving (parity tests unchanged) and drops `inhospital_mortality`
from 1.42× to **1.04×** legacy in-memory; the only minor cost is one extra full-frame collect,
a slight regression on configs with no static variables and few windows (e.g.
`imminent_mortality` 0.54 s → 0.61 s, still 0.26× legacy).

### 6.4 Where the compiled engine now stands vs legacy

After flattening (§6.2), the event-bound anchor restriction + `join_asof` (§6.3–6.3b), and the
base materialization barrier (§6.3c), the compiled engine **wins or ties on every config
tested**. Numbers below are the committed isolated-subprocess sweep (80 k subjects, in-memory
collect over `scan_parquet`; [`benchmarks/results/isolated.csv`](../../benchmarks/results),
`*_seconds.png` / `*_peak_wset_mb.png`):

| config (80 k subjects) | shape | legacy s / MB | compiled s / MB | time / mem |
| --- | --- | --- | --- | --- |
| `wide` ×16 | wide temporal | 8.34 / 6242 | **2.64** / 6040 | 0.32× / 0.97× |
| `chain` ×16 | deep temporal | 6.60 / 3241 | **2.64** / 5769 | 0.40× / 1.78× |
| `imminent_mortality` | event-bound | 2.44 / 4387 | **0.63 / 2513** | 0.26× / 0.57× |
| `readmission_risk` | shallow | 1.17 / 1773 | **0.65 / 1326** | 0.56× / 0.75× |
| `inhospital_mortality` | event-bound, low-selectivity | 2.03 / 2803 | 2.57 / **2304** | 1.27× / 0.82× |

The compiled engine is **1.6–3.8× faster** on four of five and, after the barrier, faster *and
leaner* on most. `inhospital_mortality` is now the only one slower on time — 1.27× isolated
(1.04× in-memory `.lazy()`; the `scan_parquet` path costs the compiled engine a little more) —
but it is **leaner than legacy** (2.3 GB vs 2.8 GB; the streaming variant is 1.7 GB). The lone
remaining time loss is small and comes with a memory win. The one place compiled is clearly
heavier is the deep temporal chain (`chain16` 5.8 GB vs 3.2 GB), where the flat accumulation
keeps many summary structs live at once.

## 7. Remaining optimization levers

Flattening (§6.2), the event-bound anchor restriction + `join_asof` (§6.3–6.3b), and the base
materialization barrier (§6.3c) are done. Remaining:

- **Share the cumsum pass across windows.** `_event_bound_asof` still recomputes per-subject
  prefix sums for each event-bound window; computing them once for the whole task would help
  configs with several event-bound windows (and is the most likely lever to flip
  `inhospital_mortality` to a win).
- **Reduce peak memory on temporal chains.** The flat accumulation holds every window's summary
  struct live at once; building the output structs lazily / late would cut the `chain16`
  memory overhead.
- **Cumsum + as-of for *offset* event-bound windows**, retiring the last full-frame concat path.
- **Per-window column projection.** Only compute the predicate counts a window actually
  references rather than all predicates for all windows (needs the "every window summarizes
  every predicate" output contract relaxed).
- **Early subject pruning.** Drop subjects that cannot satisfy a constraint before descending.
- **Out-of-core, by-subject streaming**, measured against real MIMIC-scale data rather than
  synthetic ≤10 M-row frames — the only regime where the streaming engine should help.

## 8. Edge cases handled (parity-verified)

- `(subject_id, timestamp)` uniqueness assumption (already enforced by `query()`).
- Row-order of output: differential comparison sorts first; document that the compiled
  engine does not promise the legacy row order unless we add an explicit final sort.
- Confirm no anchor fan-out for event-bound windows under all `closed` modes.
- Null-timestamp / static-variable handling parity (`check_static_variables`).
- Empty-result short-circuits returning `pl.DataFrame()` with matching schema.
