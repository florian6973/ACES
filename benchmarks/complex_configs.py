"""Parametric *complex* task configs (many windows) for stressing the engines.

The bundled sample configs only have 3-4 windows, which doesn't expose how an engine
scales with tree size. These builders produce configs with an arbitrary number of windows
in two shapes:

- ``chain``: W temporal windows chained end-to-start off the trigger (a deep tree).
- ``wide``: W temporal windows all hanging directly off the trigger (a bushy tree).

Both keep predicates simple (a handful of plain codes) so the only thing that varies is
the window count.
"""

from __future__ import annotations

from aces.config import EventConfig, PlainPredicateConfig, TaskExtractorConfig, WindowConfig

_PREDICATES = {
    "a": PlainPredicateConfig("a"),
    "b": PlainPredicateConfig("b"),
    "c": PlainPredicateConfig("c"),
}


def chain_config(n_windows: int, constrain: bool = False) -> TaskExtractorConfig:
    """W temporal windows, each starting where the previous ended (+1h steps)."""
    windows: dict[str, WindowConfig] = {}
    prev = "trigger"
    for i in range(n_windows):
        has = {"b": "(None, 1000000)"} if constrain else {}
        windows[f"w{i}"] = WindowConfig(
            start=prev,
            end="start + 1h",
            start_inclusive=False,
            end_inclusive=True,
            has=has,
        )
        prev = f"w{i}.end"
    return TaskExtractorConfig(predicates=dict(_PREDICATES), trigger=EventConfig("a"), windows=windows)


def wide_config(n_windows: int, constrain: bool = False) -> TaskExtractorConfig:
    """W temporal windows all anchored directly on the trigger (+(i+1)h each)."""
    windows: dict[str, WindowConfig] = {}
    for i in range(n_windows):
        has = {"b": "(None, 1000000)"} if constrain else {}
        windows[f"w{i}"] = WindowConfig(
            start="trigger",
            end=f"start + {i + 1}h",
            start_inclusive=False,
            end_inclusive=True,
            has=has,
        )
    return TaskExtractorConfig(predicates=dict(_PREDICATES), trigger=EventConfig("a"), windows=windows)


BUILDERS = {"chain": chain_config, "wide": wide_config}
