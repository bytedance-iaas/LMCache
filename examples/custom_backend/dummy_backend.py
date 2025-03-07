import asyncio
from concurrent.futures import Future
from typing import Optional

import torch

from lmcache.utils import CacheEngineKey
from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.lookup_server import LookupServerInterface
from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj)
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface

from lmcache.logging import init_logger

logger = init_logger(__name__)


class DummyBackend(StorageBackendInterface):

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: MemoryAllocatorInterface,
        dst_device: str = "cuda",
        lookup_server: Optional[LookupServerInterface] = None,
    ):
        """
        Initialize the storage backend. 

        :param dst_device: the device where the blocking retrieved KV is stored,
            could be either "cpu", "cuda", or "cuda:0", "cuda:1", etc.

        :raise: RuntimeError if the device is not valid
        """
        try:
            torch.device(dst_device)
        except RuntimeError:
            raise

        self.dst_device = dst_device
        self.memory_allocator = memory_allocator

    def contains(self, key: CacheEngineKey) -> bool:
        logger.info(f"contain {key} in dummy backend")
        return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        logger.info(f"exist put tasks {key} in dummy backend")
        return False

    def submit_put_task(self, key: CacheEngineKey,
                        obj: MemoryObj) -> Optional[Future]:
        logger.info(f"submit put task {key} in dummy backend")
        return None

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        logger.info(f"submit prefetch tast {key} in dummy backend")
        return None

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        logger.info(f"get blocking {key} in dummy backend")
        return None

    def close(self, ) -> None:
        pass
