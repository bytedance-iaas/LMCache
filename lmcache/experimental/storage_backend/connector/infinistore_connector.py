import asyncio
import ctypes
from typing import List, Optional, Union, no_type_check

import infinistore
from lmcache.experimental.memory_management import MemoryFormat

from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj)
# reuse
from lmcache.experimental.protocol import RedisMetadata
from lmcache.experimental.storage_backend.connector.base_connector import \
    RemoteConnector
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

import time

logger = init_logger(__name__)

METADATA_BYTES_LEN = 28


def _get_ptr(mv: Union[bytearray, memoryview]) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(mv))


class InfinistoreConnector(RemoteConnector):

    def __init__(self, host: str, port: int, dev_name,
                 loop: asyncio.AbstractEventLoop,
                 memory_allocator: MemoryAllocatorInterface):
        config = infinistore.ClientConfig(
            host_addr=host,
            service_port=port,
            log_level="info",
            connection_type=infinistore.TYPE_RDMA,
            ib_port=1,
            link_type=infinistore.LINK_ETHERNET,
            dev_name=dev_name,
        )

        self.rdma_conn = infinistore.InfinityConnection(config)

        self.memory_allocator = memory_allocator
        self.loop = loop
        self.rdma_conn.connect()

        # allocate 4KB buffer for RDMA read
        self.buffer_size = 4 << 10
        self.buffer = bytearray(self.buffer_size)
        self.rdma_conn.register_mr(_get_ptr(self.buffer), self.buffer_size)

    async def exists(self, key: CacheEngineKey) -> bool:

        def blocking_io():
            return self.rdma_conn.check_exist(key.to_string() + "metadata")

        return await self.loop.run_in_executor(None, blocking_io)

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        # from remote_pdb import set_trace
        # set_trace()
        logger.info(f"getting key: {key_str}")

        # count = 5
        # while count > 0:
        #     try:
        #         await self.rdma_conn.read_cache_single_async(
        #             key_str + "metadata", _get_ptr(self.buffer), len(self.buffer))
        #     except infinistore.lib.InfiniStoreKeyNotFound:
        #         logger.warning("get metadata failed: InfiniStoreKeyNotFound")
        #         count -= 1
        #         asyncio.sleep(0.5)
        #         continue
        #     except infinistore.lib.InfiniStoreKeyNotCommited:
        #         logger.warning("get metadata failed: InfiniStoreKeyNotCommited")
        #         count -= 1
        #         asyncio.sleep(0.5)
        #         continue
        #     except:
        #         return None 
        #     break
        # if count == 0:
        #     logger.warning("can't get key metadata")
        #     return None

        try:
            await self.rdma_conn.read_cache_single_async(
                key_str + "metadata", _get_ptr(self.buffer), len(self.buffer))
        except infinistore.lib.InfiniStoreKeyNotFound:
            logger.warning("get metadata failed: InfiniStoreKeyNotFound")
            return None

        metadata = RedisMetadata.deserialize(self.buffer[:METADATA_BYTES_LEN])

        memory_obj = self.memory_allocator.allocate(
            metadata.shape,
            metadata.dtype,
            metadata.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        # TODO: we could have memory allocator which pre-allocate
        # and register RDMA memory.
        # register memory is a heavy operation, so we should avoid it.
        
        ptr = None
        if metadata.fmt == MemoryFormat.BINARY_BUFFER:
            kv_bytes = bytes(memory_obj.get_size())
            pointer = ctypes.cast(ctypes.c_char_p(kv_bytes),
                              ctypes.POINTER(ctypes.c_char))
            ptr = ctypes.addressof(pointer.contents)
        elif metadata.fmt == MemoryFormat.KV_BLOB:
            kv_chunk = memory_obj.tensor
            # ptr = _get_ptr(memory_obj.byte_array)
            # ptr = ctypes.addressof(ctypes.c_ubyte.from_buffer(memory_obj.byte_array))
            ptr = kv_chunk.data_ptr()
        else:
            logger.info(f"Unsupported memory format: {metadata.fmt}")
        assert ptr is not None            
        size = memory_obj.get_size()
        logger.info(f"tensor: {memory_obj.tensor[0,0,0,0:10]}")
        logger.info(f"size: {size}, key: {key_str}")
        # logger.info(f"size: {size}, tensor")

        # from remote_pdb import set_trace
        # set_trace()
        await self.loop.run_in_executor(None, self.rdma_conn.register_mr, ptr,
                                        size)

        # count = 5
        # while count > 0:
        #     try:
        #         await self.rdma_conn.read_cache_single_async(
        #             key_str + "kv_bytes", ptr, size)
        #     except infinistore.lib.InfiniStoreKeyNotFound:
        #         logger.warning("get kv_byte failed: InfiniStoreKeyNotFound")
        #         count -= 1
        #         asyncio.sleep(0.5)
        #         continue
        #     except infinistore.lib.InfiniStoreKeyNotCommited:
        #         logger.warning("get kv_byte failed: InfiniStoreKeyNotCommited")
        #         count -= 1
        #         asyncio.sleep(0.5)
        #         continue
        #     except:
        #         return None
        #     break
        # if count == 0:
        #     logger.warning("can't get key kv_bytes")
        #     return None

        try:
            await self.rdma_conn.read_cache_single_async(
                key_str + "kv_bytes", ptr, size)
        except infinistore.lib.InfiniStoreKeyNotFound:
            logger.warning("get kv_byte failed: InfiniStoreKeyNotFound")         
            return None   

        if metadata.fmt == MemoryFormat.BINARY_BUFFER:
            view = memoryview(memory_obj.byte_array)
            view[:metadata.length] = kv_bytes

        # logger.info(f"size: {size}, key: {key_str}, tensor: {kv_chunk[0,0,0,0:10]}")
        logger.info(f"tensor: {memory_obj.tensor[0,0,0,0:10]}")
        logger.info(f"get key: {key_str} done")
        return memory_obj

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        key_str = key.to_string()
        logger.info(f"putting key: {key_str}")

        # from remote_pdb import set_trace
        # set_trace()

        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        metadata_bytes = RedisMetadata(len(kv_bytes), kv_shape, kv_dtype,
                                       memory_format).serialize()

        # not likely to happen
        assert len(metadata_bytes
                   ) <= self.buffer_size, "metadata size exceeds buffer size"

        # copy metadata to self.buffer
        self.buffer[:len(metadata_bytes)] = metadata_bytes

        await self.rdma_conn.rdma_write_cache_single_async(
            key_str + "metadata", _get_ptr(self.buffer),
            len(self.buffer))
        # from remote_pdb import set_trace
        # set_trace()
        ptr = None
        # memory_obj.byte_array is bytes
        if memory_format == MemoryFormat.BINARY_BUFFER:
            pointer = ctypes.cast(memory_obj.byte_array,
                              ctypes.POINTER(ctypes.c_char))
            ptr = ctypes.addressof(pointer.contents)
        # memory_obj.byte_array is memoryview
        elif memory_format == MemoryFormat.KV_BLOB:
            kv_chunk = memory_obj.tensor
            # ptr = _get_ptr(memory_obj.byte_array)
            # ptr = ctypes.addressof(ctypes.c_ubyte.from_buffer(memory_obj.byte_array))
            ptr = kv_chunk.data_ptr()
        else:
            logger.info(f"Unsupported memory format: {memory_format}")
        assert ptr is not None
        size = memory_obj.get_size()
        # logger.info(f"size: {size}, key: {key_str}, tensor: {kv_chunk[0,0,0,0:10]}")
        logger.info(f"size: {size}, key: {key_str}, tensor: {memory_obj.tensor[0,0,0,0:10]}")

        t1 = time.time()
        await self.loop.run_in_executor(None, self.rdma_conn.register_mr, ptr,
                                        size)
        t2 = time.time()
        logger.info(f"register_mr duration: {t2-t1}")        
        await self.rdma_conn.rdma_write_cache_single_async(
            key_str + "kv_bytes", ptr, size)

        t3 = time.time()
        logger.info(f"rdma_write_cache_single_async duration: {t3-t2}")        
        logger.info(f"put key: {key.to_string()} done")
        self.memory_allocator.ref_count_down(memory_obj)

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        self.rdma_conn.close()
        logger.info("Closed the infinistore connection")
