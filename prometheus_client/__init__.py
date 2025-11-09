from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


@dataclass
class _MetricSample:
    value: float = 0.0
    count: float = 0.0


class _Metric:
    def __init__(
        self, name: str, documentation: str, *, metric_type: str, labelnames: tuple[str, ...] = ()
    ) -> None:
        self.name = name
        self.documentation = documentation
        self.metric_type = metric_type
        self._labelnames = labelnames
        self._samples: dict[tuple[Any, ...], _MetricSample] = {}

    def labels(self, *args: Any, **kwargs: Any) -> _MetricChild:
        if args and kwargs:
            raise TypeError("Cannot mix args and kwargs for labels")
        if kwargs:
            label_values = tuple(kwargs.get(label, "") for label in self._labelnames)
        else:
            label_values = tuple(args)
        if len(label_values) != len(self._labelnames):
            raise ValueError("Incorrect number of labels provided")
        key = label_values
        sample = self._samples.setdefault(key, _MetricSample())
        return _MetricChild(self, key, sample)

    def inc(self, amount: float = 1.0) -> None:
        self.labels().inc(amount)

    def observe(self, value: float) -> None:
        self.labels().observe(value)

    def set(self, value: float) -> None:
        self.labels().set(value)

    def _render_samples(self) -> list[str]:
        lines: list[str] = [
            f"# HELP {self.name} {self.documentation}",
            f"# TYPE {self.name} {self.metric_type}",
        ]
        for label_values, sample in self._samples.items():
            if self.metric_type == "histogram":
                label_pairs = ",".join(
                    f'{label}="{value}"'
                    for label, value in zip(self._labelnames, label_values, strict=True)
                )
                prefix = f"{self.name}_"
                suffix = f"{{{label_pairs}}}" if label_pairs else ""
                lines.append(f"{prefix}sum{suffix} {sample.value}")
                lines.append(f"{prefix}count{suffix} {sample.count}")
            else:
                labels = ",".join(
                    f'{label}="{value}"'
                    for label, value in zip(self._labelnames, label_values, strict=True)
                )
                rendered = f"{self.name}"
                if labels:
                    rendered += f"{{{labels}}}"
                lines.append(f"{rendered} {sample.value}")
        if not self._samples:
            if self.metric_type == "histogram":
                lines.append(f"{self.name}_sum 0.0")
                lines.append(f"{self.name}_count 0.0")
            else:
                lines.append(f"{self.name} 0.0")
        return lines


class _MetricChild:
    def __init__(self, parent: _Metric, key: tuple[Any, ...], sample: _MetricSample) -> None:
        self._parent = parent
        self._key = key
        self._sample = sample

    def inc(self, amount: float = 1.0) -> None:
        self._sample.value += amount
        self._sample.count += amount

    def observe(self, value: float) -> None:
        self._sample.value += value
        self._sample.count += 1.0

    def set(self, value: float) -> None:
        self._sample.value = value


_REGISTRY: list[_Metric] = []


def _create_metric(
    name: str, documentation: str, metric_type: str, labelnames: tuple[str, ...]
) -> _Metric:
    metric = _Metric(name, documentation, metric_type=metric_type, labelnames=labelnames)
    _REGISTRY.append(metric)
    return metric


def Counter(
    name: str, documentation: str, labelnames: tuple[str, ...] | list[str] = ()
):  # noqa: N802
    return _create_metric(name, documentation, "counter", tuple(labelnames))


def Gauge(
    name: str, documentation: str, labelnames: tuple[str, ...] | list[str] = ()
):  # noqa: N802
    return _create_metric(name, documentation, "gauge", tuple(labelnames))


def Histogram(
    name: str, documentation: str, labelnames: tuple[str, ...] | list[str] = ()
):  # noqa: N802
    return _create_metric(name, documentation, "histogram", tuple(labelnames))


def ProcessCollector(*args: Any, **kwargs: Any) -> None:  # noqa: D401
    """No-op collector placeholder for compatibility."""


def PlatformCollector(*args: Any, **kwargs: Any) -> None:  # noqa: D401
    """No-op collector placeholder for compatibility."""


def GCCollector(*args: Any, **kwargs: Any) -> None:  # noqa: D401
    """No-op collector placeholder for compatibility."""


def generate_latest() -> bytes:
    lines: list[str] = []
    for metric in _REGISTRY:
        lines.extend(metric._render_samples())
    return "\n".join(lines).encode("utf-8")
