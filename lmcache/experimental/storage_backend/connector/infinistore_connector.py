import asyncio
import ctypes
from typing import List, Optional, Union, no_type_check

import infinistore
import torch
import operator
import numpy as np


from functools import reduce


import time

from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryFormat, MemoryObj, TensorMemoryObj, CopyLessMemoryObj)
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

        self.send_buffers = []
        self.recv_buffers = []
        self.send_queue: asyncio.Queue[int] = asyncio.Queue(
            maxsize=MAX_BUFFER_CNT)
        self.recv_queue: asyncio.Queue[int] = asyncio.Queue(
            maxsize=MAX_BUFFER_CNT)

        self.get_sum = 0
        self.put_sum = 0

        # 1KB
        self.meta_buffer_size = 1 << 10   
        
        # 40MB
        self.buffer_size = 40 << 20
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


    def get_tensor_nbytes(self, dtype: torch.dtype, shape: tuple):
        dtype_to_size = {
            torch.float64: 8,
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int64:   8,
            torch.int32:   4,
            torch.int16:   2,
            torch.int8:    1,
            torch.uint8:   1,
            torch.bool:    1,
            torch.complex64: 8,  
            torch.complex128:16,
        }

        if dtype not in dtype_to_size:
            raise ValueError(f"Unsupported dtype: {dtype}")

        element_size = dtype_to_size[dtype]
        numel = reduce(operator.mul, shape, 1)
        total_bytes = numel * element_size

        return total_bytes

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        self.get_sum += 1
        logger.debug(f"getting key: {key_str}, get_sum {self.get_sum}")

        t0 = time.perf_counter()
        try:
            buf_idx = await self.recv_queue.get()
            buffer = self.recv_buffers[buf_idx]
            await self.rdma_conn.rdma_read_cache_async(
                [(key_str + "metadata", 0)], METADATA_BYTES_LEN, _get_ptr(buffer))

            metadata = RedisMetadata.deserialize(buffer)

        finally:
            self.recv_queue.put_nowait(buf_idx)

        print(f"meta get time {time.perf_counter() - t0}")

        # memory_obj = self.memory_allocator.allocate(
        #     metadata.shape,
        #     metadata.dtype,
        #     metadata.fmt,
        # )

        # memory_obj = TensorMemoryObj(
        #     torch.empty(metadata.shape, dtype=metadata.dtype, device='cpu'),
        #     metadata=metadata,
        # )

        size = self.get_tensor_nbytes(metadata.dtype, metadata.shape)

        buf_idx = await self.recv_queue.get()
        buffer = self.recv_buffers[buf_idx]
        try:
            # await self.loop.run_in_executor(None, self.rdma_conn.register_mr,
            #                                 ptr, size)
            # await self.rdma_conn.rdma_read_cache_async(
            #     [(key_str + "kv_bytes", 0)], size, ptr)
            t1 = time.perf_counter()
            await self.rdma_conn.rdma_read_cache_async(
                [(key_str + "kv_bytes", 0)], size, _get_ptr(buffer))
            print(f"core get time {time.perf_counter() - t1}")
        except Exception as e:
            logger.warning(f"get kv_bytes failed: {e}")
            return None
        # finally:
        #     await self.recv_queue.put(buf_idx)

        def callback():
            self.recv_queue.put_nowait(buf_idx)

        t2 = time.perf_counter()
        num_elements = reduce(operator.mul, metadata.shape)
        temp_tensor = torch.frombuffer(buffer, dtype=metadata.dtype, offset=0, count=num_elements).reshape(metadata.shape)
        print(f"memory copy time {time.perf_counter() - t2}")

        # if metadata.fmt == MemoryFormat.BINARY_BUFFER:
        #     view = memoryview(memory_obj.byte_array)
        #     view[:metadata.length] = buffer[:size]
    
        logger.debug(f"get key: {key_str} done")

        memory_obj = CopyLessMemoryObj(
            raw_data=temp_tensor,
            metadata=metadata,
            callback=callback
        )
        return memory_obj

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):


        self.put_sum += 0
        t0 = time.perf_counter()

        # TODO(Jiayi): The following code is ugly.
        # Please use a function like `memory_obj.to_meta()`.
        key_str = key.to_string()
        #logger.debug(f"putting key: {key_str}")

        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()




        # assert len(metadata_bytes
        #            ) <= self.buffer_size, "metadata size exceeds buffer size"

        buf_idx = await self.send_queue.get()
        buffer = self.send_buffers[buf_idx]

        RedisMetadata(len(kv_bytes), kv_shape, kv_dtype,
                                       memory_format).serialize_into(buffer)

        # metadata_bytes = RedisMetadata(len(kv_bytes), kv_shape, kv_dtype, memory_format).serialize()
        # buffer[:len(metadata_bytes)] = metadata_bytes

        # src = np.frombuffer(metadata_bytes)
        # dest = np.frombuffer(buffer)
        # dest[:len(src)] = src


        try:
            t1 = time.perf_counter()

            await self.rdma_conn.rdma_write_cache_async(
                [(key_str + "metadata", 0)], METADATA_BYTES_LEN, _get_ptr(buffer))

            logger.debug(f"metadata put time {time.perf_counter() - t1}")

        except Exception as e:
            logger.warning(
                f"exception happens in rdma_write_cache_async metadata {e}")
            return
        # finally:
        #     self.send_queue.put_nowait(buf_idx)

        # ptr = None

        # buf_idx = await self.send_queue.get()
        # buffer = self.send_buffers[buf_idx]
        
        assert len(kv_bytes) <= self.buffer_size


        # buffer[:len(kv_bytes)] = kv_bytes

        t2 = time.perf_counter()
        src = np.frombuffer(kv_bytes)
        dest = np.frombuffer(buffer)
        dest[:len(src)] = src
        # buffer[:len(kv_bytes)] = kv_bytes
        logger.debug(f"copy takes {time.perf_counter()- t2}")



        # memory_obj.byte_array is bytes
        # if memory_format == MemoryFormat.BINARY_BUFFER:
        #     pointer = ctypes.cast(ctypes.c_char_p(memory_obj.byte_array),
        #                           ctypes.POINTER(ctypes.c_char))
        #     ptr = ctypes.addressof(pointer.contents)
        # # memory_obj.byte_array is memoryview
        # elif memory_format == MemoryFormat.KV_BLOB:
        #     kv_chunk = memory_obj.tensor
        #     if kv_chunk is not None:
        #         ptr = kv_chunk.data_ptr()
        # else:
        #     logger.warning(f"Unsupported memory format: {memory_format}")
        # assert ptr is not None
        size = memory_obj.get_size()

        try:
            # await self.loop.run_in_executor(None, self.rdma_conn.register_mr,
            #                                 ptr, size)
            # await self.rdma_conn.rdma_write_cache_async(
            #     [(key_str + "kv_bytes", 0)], size, ptr)
            t4 = time.perf_counter()
            await self.rdma_conn.rdma_write_cache_async(
                [(key_str + "kv_bytes", 0)], size, _get_ptr(buffer)
            )
            logger.debug(f"kvcache put time {time.perf_counter() - t4}, size {size/1e6:.4f} MB")


        except Exception as e:
            logger.warning(
                f"exception happens in rdma_write_cache_async kv_bytes {e}")
            return
        finally:
            await self.send_queue.put(buf_idx)

        #logger.debug(f"put key: {key.to_string()} done")
        self.memory_allocator.ref_count_down(memory_obj)
        logger.debug(f"all infinistore put time {time.perf_counter() - t0}, sum {self.put_sum}")


    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        self.rdma_conn.close()
        logger.info("Closed the infinistore connection")
