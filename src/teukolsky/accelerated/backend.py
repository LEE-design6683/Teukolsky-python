from __future__ import annotations

from typing import Any

import torch


def dcu_status(device_id: int = 0) -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    status = {
        "backend": "DCU",
        "torch_version": torch.__version__,
        "available": available,
        "device_count": count,
        "device_id": device_id,
        "device_name": None,
    }
    if available:
        if device_id < 0 or device_id >= count:
            raise RuntimeError(
                f"requested DCU device_id={device_id}, but only {count} CUDA/ROCm device(s) are visible"
            )
        status["device_name"] = torch.cuda.get_device_name(device_id)
        status["device"] = f"cuda:{device_id}"
    return status


def require_dcu(device_id: int = 0) -> dict[str, Any]:
    status = dcu_status(device_id)
    if not status["available"]:
        raise RuntimeError(
            "DCU backend is not available in this session: torch.cuda.is_available() is False"
        )
    return status
