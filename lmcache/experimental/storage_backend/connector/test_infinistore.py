from lmcache.experimental.storage_backend.connector import InfinistoreConnector
import asyncio
from lmcache.experimental.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    TensorMemoryObj
)
from lmcache.experimental.protocol import RedisMetadata
from lmcache.utils import CacheEngineKey

import torch


async def run():
    loop = asyncio.get_event_loop()
    conn = InfinistoreConnector("127.0.0.1", 12345, "mlx5_3", loop, None)

    tensor = torch.rand(4,4,4,4)
    metadata= RedisMetadata(4*4*4*4*4, (4,4,4,4), torch.float32, MemoryFormat.KV_BLOB)
    memory_obj = TensorMemoryObj(
        tensor,
        metadata=metadata,
    )


    key = CacheEngineKey(
        fmt="vllm",
        model_name="none",
        world_size=1,
        worker_id=0,
        chunk_hash="asdfasdfasdf"
    )

    print(memory_obj.get_shape())
    print(memory_obj.raw_data[0][0][0][0])

    await conn.put(key, memory_obj)






    output = await conn.get(key)
    print(output.get_shape())
    print(output.raw_data[0][0][0][0])
    await conn.close()


asyncio.run(run())