from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_percent: float
    ram_used_bytes: int
    ram_total_bytes: int
    disk_free_bytes: int
    gpu_utilization_percent: float | None = None
    gpu_memory_used_bytes: int | None = None
    gpu_memory_total_bytes: int | None = None
    gpu_temperature_c: float | None = None
    gpu_power_w: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResourceMonitor:
    def __init__(self, output: Path, gpu_index: int = 0) -> None:
        self.output = output
        self.gpu_handle: Any | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception:
            self._pynvml = None

    def snapshot(self) -> ResourceSnapshot:
        memory = psutil.virtual_memory()
        disk = shutil.disk_usage(self.output.parent if not self.output.exists() else self.output)
        values: dict[str, Any] = {}
        if self.gpu_handle is not None and self._pynvml is not None:
            try:
                utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_memory = self._pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                values = {
                    "gpu_utilization_percent": float(utilization.gpu),
                    "gpu_memory_used_bytes": int(gpu_memory.used),
                    "gpu_memory_total_bytes": int(gpu_memory.total),
                    "gpu_temperature_c": float(
                        self._pynvml.nvmlDeviceGetTemperature(
                            self.gpu_handle, self._pynvml.NVML_TEMPERATURE_GPU
                        )
                    ),
                    "gpu_power_w": float(self._pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle))
                    / 1000.0,
                }
            except Exception:
                values = {}
        return ResourceSnapshot(
            cpu_percent=psutil.cpu_percent(interval=None),
            ram_used_bytes=int(memory.used),
            ram_total_bytes=int(memory.total),
            disk_free_bytes=int(disk.free),
            **values,
        )
