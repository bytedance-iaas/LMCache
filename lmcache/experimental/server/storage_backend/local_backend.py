import threading
from collections import OrderedDict
from typing import List, Optional
import sys

from lmcache.experimental.protocol import ClientMetaMessage
from lmcache.experimental.server.storage_backend.abstract_backend import \
    LMSBackendInterface
from lmcache.experimental.storage_backend.evictor.lru_evictor import LRUEvictor
from lmcache.experimental.storage_backend.evictor.base_evictor import PutStatus
from lmcache.experimental.server.utils import LMSMemoryObj
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate


logger = init_logger(__name__)


class LMSLocalBackend(LMSBackendInterface):

    def __init__(self, ):
        self.dict: OrderedDict[CacheEngineKey, LMSMemoryObj] = OrderedDict()

        self.lock = threading.Lock()
        self.evictor = LRUEvictor(10.0)

        # TODO(Jiayi): please add evictor

    # TODO
    def list_keys(self) -> List[CacheEngineKey]:
        with self.lock:
            return list(self.dict.keys())

    def contains(
        self,
        key: CacheEngineKey,
    ) -> bool:

        with self.lock:
            return key in self.dict

    # TODO
    def remove(
        self,
        key: CacheEngineKey,
    ) -> None:

        with self.lock:
            self.dict.pop(key)

    def put(
        self,
        client_meta: ClientMetaMessage,
        kv_chunk_bytes: bytearray,
    ) -> None:

        with self.lock:
            self.dict[client_meta.key] = LMSMemoryObj(
                kv_chunk_bytes,
                client_meta.length,
                client_meta.fmt,
                client_meta.dtype,
                client_meta.shape,
            )
            keys, status = self.evictor.update_on_put(self.dict, sys.getsizeof(kv_chunk_bytes))
            if status == PutStatus.ILLEGAL:
                return
            for key in keys:
                self.dict.pop(key)

    @_lmcache_nvtx_annotate
    def get(
        self,
        key: CacheEngineKey,
    ) -> Optional[LMSMemoryObj]:

        with self.lock:
            res = self.dict.get(key, None)
            if res is not None:
                self.evictor.update_on_get(self.dict)
            return res

    def close(self):
        pass


# TODO(Jiayi): please implement the remote disk backend
#class LMSLocalDiskBackend(LMSBackendInterface):
#    pass
