## Configuration Language Specification

This document specifies the configuration language for the automatic extraction of task dataframes and cohorts
from structured EHR data organized either via the [MEDS](https://github.com/Medical-Event-Data-Standard/meds)
format (recommended) or the [ESGPT](https://eventstreamml.readthedocs.io/en/latest/) format. This extraction
system works by defining a configuration object that details the underlying concepts, inclusion/exclusion, and
labeling criteria for the cohort/task to be extracted, then using a recursive algorithm to identify all
realizations of valid patient time-ranges of data that satisfy those constraints from the raw data. For more
details on the recursive algorithm, see [Algorithm Design](https://eventstreamaces.readthedocs.io/en/latest/technical.html#algorithm-design).

As indicated above, these cohorts are specified through a combination of concepts (realized as event
_predicate_ functions, _aka_ "predicates") which are _dataset specific_ and inclusion/exclusion/labeling
criteria which, conditioned on a set of predicate definitions, are _dataset agnostic_.

Predicates are currently limited to "count" predicates, which are predicates that count the number of times a
boolean condition is satisfied over a given time window, which can either be a single timestamp, thus tracking
whether how many observations there were that satisfied the boolean condition in that event (_aka_ at that
timestamp) or over 1-dimensional windows. In the future, predicates may expand to include other notions of
functional characterization, such as tracking the average/min/max value a concept takes on over a time-period,
etc.

Constraints are specified in terms of time-points that can be bounded by events that satisfy predicates or
temporal relationships on said events. The windows between these time-points can then either be constrained to
contain events that satisfy certain aggregation functions over predicates for these time frames.

______________________________________________________________________

In the machine form used by ACES, the configuration file consists of three parts:

- `predicates`, stored as a dictionary from string predicate names (which must be unique) to
  {py:class}`aces.config.PlainPredicateConfig` objects, which store raw predicates with no dependencies on other predicates,
  {py:class}`aces.config.DerivedPredicateConfig` objects, which store boolean combinations of other predicates, or
  the relational {py:class}`aces.config.DuringPredicateConfig` / {py:class}`aces.config.WithinPredicateConfig`
  objects, which define a predicate from a temporal relation to other events.
- `trigger`, stored as a string to `EventConfig`
- `windows`, stored as a dictionary from string window names (which must be unique) to {py:class}`aces.config.WindowConfig`
  objects.

Below, we will detail each of these configuration objects.

______________________________________________________________________

### Predicates: `PlainPredicateConfig` and `DerivedPredicateConfig`

#### {py:class}`aces.config.PlainPredicateConfig`: Configuration of Predicates that can be Computed Directly from Raw Data

These configs consist of the following four fields:

- `code`: The string expression for the code object that is relevant for this predicate. An
  observation will only satisfy this predicate if there is an occurrence of this code in the observation.
  The field can additionally be a dictionary with either a `regex` key and the value being a regular
  expression (satisfied if the regular expression evaluates to True), or a `any` key and the value being a
  list of strings (satisfied if there is an occurrence for any code in the list).

  > [!NOTE]
  > Each individual definition of `PlainPredicateConfig` and `code` will generate a separate predicate
  > column. Thus, for memory optimization, it is strongly recommended to match multiple values using either
  > the List of Values or Regular Expression formats whenever possible.

- `value_min`: If specified, an observation will only satisfy this predicate if the occurrence of the
  underlying `code` with a reported numerical value that is either greater than or greater than or equal to
  `value_min` (with these options being decided on the basis of `value_min_inclusive`, where
  `value_min_inclusive=True` indicating that an observation satisfies this predicate if its value is greater
  than or equal to `value_min`, and `value_min_inclusive=False` indicating a greater than but not equal to
  will be used).

- `value_max`: If specified, an observation will only satisfy this predicate if the occurrence of the
  underlying `code` with a reported numerical value that is either less than or less than or equal to
  `value_max` (with these options being decided on the basis of `value_max_inclusive`, where
  `value_max_inclusive=True` indicating that an observation satisfies this predicate if its value is less
  than or equal to `value_max`, and `value_max_inclusive=False` indicating a less than but not equal to
  will be used).

- `value_min_inclusive`: See `value_min`

- `value_max_inclusive`: See `value_max`

- `other_cols`: This optional field accepts a 1-to-1 dictionary of column names to column values, and can be
  used to specify further constraints on other columns (ie., not `code`) for this predicate.

A given observation will be gauged to satisfy or fail to satisfy this predicate in one of two ways, depending
on its source format.

1. If the source data is in [MEDS](https://github.com/Medical-Event-Data-Standard/meds) format
   (recommended), then the `code` will be checked directly against MEDS' `code` field and the `value_min`
   and `value_max` constraints will be compared against MEDS' `numeric_value` field.

   > [!NOTE]
   > This syntax does not currently support defining predicates that also rely on matching other, optional
   > fields in the MEDS syntax; if this is a desired feature for you, please let us know by filing a GitHub
   > issue or pull request or upvoting any existing issue/PR that requests/implements this feature, and we
   > will add support for this capability.

2. If the source data is in [ESGPT](https://eventstreamml.readthedocs.io/en/latest/) format, then the
   `code` will be interpreted in the following manner:
   a. If the code contains a `"//"`, it will be interpreted as being a two element list joined by the
   `"//"` character, with the first element specifying the name of the ESGPT measurement under
   consideration, which should either be of the multi-label classification or multivariate regression
   type, and the second element being the name of the categorical key corresponding to the code in
   question within the underlying measurement specified. If either of `value_min` and `value_max` are
   present, then this measurement must be of a multivariate regression type, and the corresponding
   `values_column` for extracting numerical observations from ESGPT's `dynamic_measurements_df` will be
   sourced from the ESGPT dataset configuration object.
   b. If the code does not contain a `"//"`, it will be interpreted as a direct measurement name that must
   be of the univariate regression type and its value, if needed, will be pulled from the corresponding
   column.

#### {py:class}`aces.config.DerivedPredicateConfig`: Configuration of Predicates that Depend on Other Predicates

These configuration objects consist of only a single string field--`expr`--which contains a limited grammar of
accepted operations that can be applied to other predicates, containing precisely the following:

- `and(pred_1_name, pred_2_name, ...)`: Asserts that all of the specified predicates must be true.
- `or(pred_1_name, pred_2_name, ...)`: Asserts that any of the specified predicates must be true.

> [!NOTE]
> Currently, `and`'s and `or`'s cannot be nested. Upon user request, we may support further advanced
> analytic operations over predicates.

#### Relational predicates: {py:class}`aces.config.DuringPredicateConfig` and {py:class}`aces.config.WithinPredicateConfig`

The predicates above are *monadic* — each is a property of a single event. Relational predicates instead
define a predicate from a **temporal relation to other events**, but the result is still materialized as an
ordinary 0/1 predicate column per event, so it can be used anywhere a predicate is accepted (any window's
`has`/`has_any`, `label`, the `trigger`, or inside a derived `expr`).

There are two relational constructors, both expressing "a point inside an interval anchored to other
event(s)", differing only in how the interval is sourced:

- **`during`** — the interval is a **reconstructed open/close pair**. `during` accepts the fields `event`
  (restricts which events this predicate can hold for), `opens` (interval-open boundary predicate), `closes`
  (interval-close boundary predicate), and `closed` (inclusivity of the reconstructed `[opens, closes]`
  interval: one of `both` (default), `left`, `right`, or `none`). `during(e)` is true iff `event(e)` holds
  **and** `e`'s timestamp lies inside an open interval, reconstructed via the **net-open count**: an event is
  "admitted" at time `τ` iff the number of `opens` at-or-before `τ` strictly exceeds the number of `closes`
  before `τ` (the exact `≤`/`<` edges follow `closed`). This is exact for non-overlapping stays; for
  nested/overlapping stays it is the "currently admitted" semantics (an unclosed `opens` keeps admitting).

  ```yaml
  predicates:
    ami_enc:
      during:
        event: ami            # restrict THIS predicate's events
        opens: ip_er_start    # interval-open boundary predicate
        closes: ip_er_end     # interval-close boundary predicate
        closed: both          # both | left | right | none  (default both)
  ```

- **`within`** — the interval is a **fixed offset window around another event**. `within` accepts the fields
  `event`, `of` (the corroborating predicate), `before`, and `after` (timedelta strings, e.g. `365d`, each
  defaulting to zero for a one-sided window). `within(e)` is true iff `event(e)` holds **and** there exists an
  `of`-event whose timestamp lies in `[e.time - before, e.time + after]`. This is a pure proximity (range)
  join, with no interval reconstruction and hence no overlap/pairing ambiguity.

  ```yaml
  predicates:
    cs4_smk:
      within:
        event: cs4            # restrict THIS predicate's events
        of: smoking           # the corroborating predicate
        before: 365d          # look this far back from each `event` occurrence
        after: 365d           # ... and this far forward (either may be 0 for one-sided)
  ```

Because the output is a plain predicate column, `has: {ami_enc: (1, None)}` ("∃ encounter-AMI in the
window"), `has: {ami_enc: (None, 0)}` ("no prior encounter-AMI"), `label: ami_enc`, and `expr: and(ami,
ami_enc)` all just work.

______________________________________________________________________

### Events: {py:class}`aces.config.EventConfig`

The event config consists of only a single field, `predicate`, which specifies the predicate that must be
observed with value greater than one to satisfy the event. There can only be one defined "event" with an
"EventConfig" in a valid configuration, and it will define the "trigger" event of the cohort.

The value of its field can be any defined predicate.

______________________________________________________________________

### Windows: {py:class}`aces.config.WindowConfig`

Windows contain a tracking `name` field, and otherwise are specified with two parts: (1) A set of four
parameters (`start`, `end`, `start_inclusive`, and `end_inclusive`) that specify the time range of the window,
and (2) a set of constraints specified through two fields, dictionary of constraints (the `has` field) that
specify the constraints that must be satisfied over the defined predicates for a possible realization of this
window to be valid.

#### Time Range Fields

##### `start` and `end`

Valid windows always progress in time from the `start` field to the `end` field. These two fields define, in
symbolic form, the relationship between the start and end time of the window. These two fields must obey the
following rules:

1. _Linkage to other windows_: Firstly, exactly one of these two fields must reference an external event, as
   specified either through the name of the trigger event or the start or end event of another window. The other
   field must either be `null`/`None`/omitted (which has a very specific meaning, to be explained shortly) or
   must reference the field that references the external event.

2. _Linkage reference language_: Secondly, for both events, regardless of whether they reference an external
   event or an internal event, that reference must be expressed in one of the following ways.

   1. `$REFERENCING = $REFERENCED + $TIME_DELTA`, `$REFERENCING = $REFERENCED - $TIME_DELTA`, etc.
      In this case, the referencing event (either the start or end of the window) will be defined as occurring
      exactly `$TIME_DELTA` either after or before the event being referenced (either the external event or the
      end or start of the window).

      > [!NOTE]
      > If `$REFERENCED` is the `start` field, then `$TIME_DELTA` must be positive, and if
      > `$REFERENCED` is the `end` field, then `$TIME_DELTA` must be negative to preserve the time ordering of
      > the window fields.

   2. `$REFERENCING = $REFERENCED -> $PREDICATE`, `$REFERENCING = $REFERENCED <- $PREDICATE`
      In this case, the referencing event will be defined as the next or previous event satisfying the
      predicate, `$PREDICATE`.

      > [!NOTE]
      > If the `$REFERENCED` is the `start` field, then the "next predicate
      > ordering" (`$REFERENCED -> $PREDICATE`) must be used, and if the `$REFERENCED` is the `end` field, then
      > the "previous predicate ordering" (`$REFERENCED <- $PREDICATE`) must be used to preserve the time
      > ordering of the window fields. These forms can lead to windows being defined as single point events, if
      > the `$REFERENCED` event itself satisfies `$PREDICATE` and the appropriate constraints are satisfied and
      > inclusive values are set.

   3. `$REFERENCING = $REFERENCED`
      In this case, the referencing event will be defined as the same event as the referenced event.

3. _`null`/`None`/omitted_: If `start` is `null`/`None`/omitted, then the window will start at the beginning of
   the patient's record. If `end` is `null`/`None`/omitted, then the window will end at the end of the patient's
   record. In either of these cases, the other field must reference an external event, per rule 1.

##### `start_inclusive` and `end_inclusive`

These two fields specify whether the start and end of the window are inclusive or exclusive, respectively.
This applies both to whether they are included in the calculation of the predicate values over the windows,
but also, in the `$REFERENCING = $REFERENCED -> $PREDICATE` and `$REFERENCING = $PREDICATE -> $REFERENCED`
cases, to which events are possible to use for valid next or prior `$PREDICATE` events. E.g., if we have that
`start_inclusive=False` and the `end` field is equal to `start -> $PREDICATE`, and it so happens that the
`start` event itself satisfies `$PREDICATE`, the fact that `start_inclusive=False` will mean that we do not
consider the `start` event itself to be a valid start to any window that ends at the same `start` event, as
its timestamp when considered as the prospective "window start timestamp" occurs "after" the effective
timestamp of itself when considered as the `$PREDICATE` event that marks the window end given that
`start_inclusive=False` and thus we will think of the window as truly starting an iota after the timestamp of
the `start` event itself.

#### Constraints Field

The constraints field is a dictionary that maps predicate names to tuples of the form `(min_valid, max_valid)`
that define the valid range the count of observations of the named predicate that must be found in a window
for it to be considered valid. Either `min_valid` or `max_valid` constraints can be `None`, in which case
those endpoints are left unconstrained. Likewise, unreferenced predicates are also left unconstrained.

> [!NOTE]
> As predicate counts are always integral, this specification does not need an additional
> inclusive/exclusive endpoint field, as one can simply increment the bound by one in the appropriate direction
> to achieve the result. Instead, this bound is always interpreted to be inclusive, so a window would satisfy
> the constraint for predicate `name` with constraint `name: (1, 2)` if the count of observations of predicate
> `name` in a window was either 1 or 2. All constraints in the dictionary must be satisfied on a window for it
> to be included.
#### Disjunctive constraints: the `has_any` field

The `has` field is a **conjunction** — every listed constraint must hold. To express a **disjunction** of
constraint blocks, use `has_any`: a list of blocks where each block is itself an ordinary (conjunctive) `has`
dictionary. The window is valid iff **at least one** block is fully satisfied.

```yaml
at_risk_entry:
  start: null
  end: trigger
  start_inclusive: true
  end_inclusive: true
  has_any:                         # OR of blocks; each block is an AND of leaf constraints
    - cs13: (1, None)              # CS1 ∪ CS3
    - cs4: (2, None)               # >= 2 CS4
    - cs4_smk: (1, None)           # CS4 corroborated by smoking
```

> [!NOTE]
> `has` (conjunction) and `has_any` (disjunction) are mutually exclusive on the same window. `has_any` adds
> exactly the one missing top-level OR connective; arbitrary nesting and negation of constraint trees remain
> out of scope. When writing blocks in YAML flow style (`- {cs13: (1, None)}`), quote the value
> (`- {cs13: "(1, None)"}`) so the comma is not parsed as a flow separator; block style (shown above) needs no
> quoting.
