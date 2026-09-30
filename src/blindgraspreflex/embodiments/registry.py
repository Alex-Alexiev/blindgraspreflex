"""Discovery of built-in and third-party embodiment plugins."""

from __future__ import annotations

from importlib.metadata import entry_points

from blindgraspreflex.embodiments.spec import EmbodimentSpec

ENTRY_POINT_GROUP = "blindgraspreflex.embodiments"


def _entry_points():
    return entry_points(group=ENTRY_POINT_GROUP)


def available() -> tuple[str, ...]:
    return tuple(sorted(ep.name for ep in _entry_points()))


def load(name: str) -> EmbodimentSpec:
    matches = [ep for ep in _entry_points() if ep.name == name]
    if not matches:
        choices = ", ".join(available()) or "none installed"
        raise KeyError(f"Unknown embodiment {name!r}; available: {choices}")
    if len(matches) > 1:
        sources = ", ".join(ep.value for ep in matches)
        raise RuntimeError(f"Multiple embodiment plugins are registered as {name!r}: {sources}")
    spec = matches[0].load()()
    if not isinstance(spec, EmbodimentSpec):
        raise TypeError(
            f"Embodiment plugin {name!r} returned {type(spec).__name__}, not EmbodimentSpec"
        )
    return spec
