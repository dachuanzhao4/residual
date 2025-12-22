from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def _is_scalar(v: Any) -> bool:
    if isinstance(v, (int, float)):
        return True
    if hasattr(v, "numel") and hasattr(v, "item"):
        try:
            return int(v.numel()) == 1
        except Exception:
            return False
    return False


def _to_float(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    return float(v.item())


@dataclass
class MetricEvent:
    timestamp: float
    step: int
    epoch: int | None
    key: str
    value: float


class CsvMetricSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", newline="")
        self._writer = csv.writer(self._fp)

        should_write_header = self.path.stat().st_size == 0
        if should_write_header:
            self._writer.writerow(["timestamp", "step", "epoch", "key", "value"])
            self._fp.flush()

    def log(self, metrics: Mapping[str, float], *, step: int, epoch: int | None = None) -> None:
        ts = time.time()
        for key, value in metrics.items():
            self._writer.writerow([ts, step, epoch, key, value])
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass


class WandbMetricSink:
    def __init__(self, *, project: str, name: str, config: Mapping[str, Any]):
        import wandb

        self._wandb = wandb
        self._wandb.init(project=project, name=name, config=dict(config))

    def log(self, metrics: Mapping[str, float], *, step: int, epoch: int | None = None) -> None:
        payload = dict(metrics)
        if epoch is not None:
            payload.setdefault("epoch", epoch)
        self._wandb.log(payload, step=step)

    def close(self) -> None:
        try:
            self._wandb.finish()
        except Exception:
            pass


class MetricLogger:
    def __init__(self, *, sinks: Sequence[Any] = ()):
        self._sinks = list(sinks)

    def log(self, metrics: Mapping[str, Any], *, step: int, epoch: int | None = None) -> None:
        scalar_metrics: dict[str, float] = {}
        for key, value in metrics.items():
            if value is None:
                continue
            if _is_scalar(value):
                scalar_metrics[key] = _to_float(value)
        if not scalar_metrics:
            return
        for sink in self._sinks:
            sink.log(scalar_metrics, step=step, epoch=epoch)

    def close(self) -> None:
        for sink in self._sinks:
            sink.close()


def build_metric_logger(
    *,
    backend: str,
    is_main_process: bool,
    log_dir: str | Path | None,
    project: str,
    run_name: str,
    config: Mapping[str, Any],
    csv_filename: str = "metrics.csv",
) -> MetricLogger:
    if not is_main_process or backend in ("none", "off", ""):
        return MetricLogger()

    backends = [b.strip().lower() for b in backend.split(",") if b.strip()]
    sinks: list[Any] = []
    if "csv" in backends:
        if log_dir is None:
            raise ValueError("csv backend requires log_dir")
        sinks.append(CsvMetricSink(Path(log_dir) / csv_filename))
    if "wandb" in backends:
        sinks.append(WandbMetricSink(project=project, name=run_name, config=config))
    return MetricLogger(sinks=sinks)

