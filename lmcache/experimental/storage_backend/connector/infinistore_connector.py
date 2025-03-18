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

logger = init_logger(__name__)

METADATA_BYTES_LEN = 28

MAX_BUFFER_CNT = 128

def _get_ptr(mv: Union[bytearray, memoryview]) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(mv))


class InfinistoreConnector(RemoteConnector):

    def __init__(self, host: str, port: int, dev_name,
                 loop: asyncio.AbstractEventLoop,
                 memory_allocator: MemoryAllocatorInterface):
        config = infinistore.ClientConfig(
            host_addr=host,
            service_port=port,
            log_level="debug",
            connection_type=infinistore.TYPE_RDMA,
            ib_port=1,
            link_type=infinistore.LINK_ETHERNET,
            dev_name=dev_name,
        )

        self.rdma_conn = infinistore.InfinityConnection(config)

        self.memory_allocator = memory_allocator
        self.loop = loop
        self.rdma_conn.connect()

        self.send_buffers = []
        self.recv_buffers = []
        self.send_queue = asyncio.Queue(maxsize=MAX_BUFFER_CNT)
        self.recv_queue = asyncio.Queue(maxsize=MAX_BUFFER_CNT)

        # 4KB buffer for send/recv metadata
        self.buffer_size = 4 << 10
        for i in range(MAX_BUFFER_CNT):
            send_buffer = bytearray(self.buffer_size)
            self.rdma_conn.register_mr(_get_ptr(send_buffer), self.buffer_size)
            self.send_buffers.append(send_buffer)
            self.send_queue.put_nowait(i)

            recv_buffer = bytearray(self.buffer_size)
            self.rdma_conn.register_mr(_get_ptr(recv_buffer), self.buffer_size)
            self.recv_buffers.append(recv_buffer)
            self.recv_queue.put_nowait(i)

    async def exists(self, key: CacheEngineKey) -> bool:

        def blocking_io():
            return self.rdma_conn.check_exist(key.to_string() + "metadata")

        return await self.loop.run_in_executor(None, blocking_io)

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        logger.info(f"getting key: {key_str}")

        try:
            buf_idx = await self.recv_queue.get()
            buffer = self.recv_buffers[buf_idx]
            await self.rdma_conn.rdma_read_cache_async(
                [(key_str + "metadata", 0)], len(buffer), _get_ptr(buffer))
            metadata = RedisMetadata.deserialize(buffer[:METADATA_BYTES_LEN])
        except:
            logger.warning("get metadata failed: InfiniStoreKeyNotFound")
            return None
        finally:
            self.recv_queue.put_nowait(buf_idx)

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
            ptr = kv_chunk.data_ptr()
        else:
            logger.info(f"Unsupported memory format: {metadata.fmt}")
        assert ptr is not None
        size = memory_obj.get_size()

        try:
            await self.loop.run_in_executor(None, self.rdma_conn.register_mr, ptr,
                                        size)
            await self.rdma_conn.rdma_read_cache_async(
                [(key_str + "kv_bytes", 0)], size, ptr)
        except:
            logger.warning("get kv_bytes failed: InfiniStoreKeyNotFound")
            return None

        if metadata.fmt == MemoryFormat.BINARY_BUFFER:
            view = memoryview(memory_obj.byte_array)
            view[:metadata.length] = kv_bytes
        logger.info(f"get key: {key_str} done")

        return memory_obj

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        # TODO(Jiayi): The following code is ugly.
        # Please use a function like `memory_obj.to_meta()`.
        key_str = key.to_string()
        logger.info(f"putting key: {key_str}")

        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        metadata_bytes = RedisMetadata(len(kv_bytes), kv_shape, kv_dtype,
                                       memory_format).serialize()

        assert len(metadata_bytes
                   ) <= self.buffer_size, "metadata size exceeds buffer size"

        buf_idx = await self.send_queue.get()
        buffer = self.send_buffers[buf_idx]
        buffer[:len(metadata_bytes)] = metadata_bytes
        try:
            await self.rdma_conn.rdma_write_cache_async(
                [(key_str + "metadata", 0)], len(buffer), _get_ptr(buffer))
        except:
            logger.warning("exception happens during rdma_write_cache_async metadata")
            return
        finally:
            self.send_queue.put_nowait(buf_idx)

        ptr = None
        # memory_obj.byte_array is bytes
        if memory_format == MemoryFormat.BINARY_BUFFER:
            pointer = ctypes.cast(memory_obj.byte_array,
                                  ctypes.POINTER(ctypes.c_char))
            ptr = ctypes.addressof(pointer.contents)
        # memory_obj.byte_array is memoryview
        elif memory_format == MemoryFormat.KV_BLOB:
            kv_chunk = memory_obj.tensor
            ptr = kv_chunk.data_ptr()
        else:
            logger.info(f"Unsupported memory format: {memory_format}")
        assert ptr is not None
        size = memory_obj.get_size()

        try:
            await self.loop.run_in_executor(None, self.rdma_conn.register_mr, ptr,
                                            size)
            await self.rdma_conn.rdma_write_cache_async(
                [(key_str + "kv_bytes", 0)], size, ptr)
        except:
            logger.warning("exception happens during register_mr and rdma_write_cache_async kv_bytes")
            return
        
        logger.info(f"put key: {key.to_string()} done")
        self.memory_allocator.ref_count_down(memory_obj)

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        self.rdma_conn.close()
        logger.info("Closed the infinistore connection")
