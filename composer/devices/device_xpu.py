# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The XPU device used for training."""

from __future__ import annotations

from typing import Any, Dict, Optional, TypeVar

import torch
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
import torch.xpu.amp
import torch.utils.data

from composer.devices.device import Device
from composer.utils import dist

__all__ = ['DeviceXPU']

T_nnModule = TypeVar('T_nnModule', bound=torch.nn.Module)


class DeviceXPU(Device):
    """An extension of :class:`~composer.devices.device.Device` for XPUs.

    Args:
        device_id (int, optional): Integer ID of a XPU device to train with. If not specified, the local rank
            of the current process is used. Default: None.
    """
    dist_backend = 'ccl'
    name = 'xpu'

    def __init__(
        self,
        device_id: Optional[int] = None,
    ):
        if not torch.xpu.is_available():
            raise ValueError('DeviceXPU cannot be created as torch.xpu is not available.')
        if device_id is None:
            device_id = dist.get_local_rank()
        self._device = torch.device(f'xpu:{device_id}')
        torch.xpu.set_device(self._device)
        assert torch.xpu.current_device() == device_id

    def module_to_device(self, module: T_nnModule) -> T_nnModule:
        return module.to(self._device)

    def tensor_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device, non_blocking=True)

    def state_dict(self) -> Dict[str, Any]:
        return {
            'rng': torch.xpu.get_rng_state(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        torch.xpu.set_rng_state(state['rng'])
