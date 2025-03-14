import asyncio
import multiprocessing
from typing import Dict, List, Optional

import torch

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.distributed_server import (
    DistributedServerInterface, NaiveDistributedServer)
from lmcache.experimental.gpu_connector import GPUConnectorInterface
from lmcache.experimental.lookup_server import (LookupServerInterface,
                                                RedisLookupServer)
from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MixedMemoryAllocator)
from lmcache.experimental.storage_backend.storage_manager import StorageManager
from lmcache.experimental.token_database import (ChunkedTokenDatabase,
                                                 TokenDatabase)
from lmcache.logging import init_logger
from lmcache.observability import LMCacheStatsLogger, LMCStatsMonitor
from lmcache.usage_context import InitializeUsageContext
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)


class CacheEngineEndSignal:
    pass


class LMCacheEngine:
    """The main class for the cache engine. 

    When storing the KV caches into the cache engine, it takes GPU KV
    caches from the serving engine and convert them into MemoryObjs that
    resides in the CPU. The MemoryObjs are then being stored into the 
    StorageBackends in an asynchronous manner.

    When retrieving the KV caches from the cache engine, it fetches the
    MemoryObjs from the StorageBackends and convert them into GPU KV caches
    by GPUConnectors specialized for the serving engine.

    It also supports prefetching the KV caches from the StorageBackends. 
    It relies on the StorageBackends to manage the requests of prefetching
    and real retrieval and avoid the conflicts.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        memory_allocator: MemoryAllocatorInterface,
        token_database: TokenDatabase,
        gpu_connector: GPUConnectorInterface,
    ):
        self.config = config
        self.metadata = metadata
        self.memory_allocator = memory_allocator
        self.token_database = token_database
        self.gpu_connector = gpu_connector

        self.enable_p2p = config.enable_p2p

        # NOTE: Unix systems use fork by default
        multiprocessing.set_start_method('spawn', force=True)

        self.lookup_server: Optional[LookupServerInterface] = None
        # TODO(Jiayi): hard-coded for now
        if self.enable_p2p:
            self.lookup_server = RedisLookupServer(config)

        self.storage_manager = StorageManager(config, metadata,
                                              self.memory_allocator,
                                              self.lookup_server)
        if self.enable_p2p:
            self.distributed_loop = asyncio.get_event_loop()
            assert self.lookup_server is not None
            self.distributed_server: DistributedServerInterface = \
                NaiveDistributedServer(self.storage_manager,
                                       self.lookup_server,
                                       self.memory_allocator,
                                       self.distributed_loop,
                                       config)
        InitializeUsageContext(config.to_original_config(), metadata)
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store(self,
              tokens: torch.Tensor,
              mask: Optional[torch.Tensor] = None,
              **kwargs) -> None:
        """Store the tokens and mask into the cache engine.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should 
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched, 
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables). 

        :raises: ValueError if the number of Falses in the mask is not a 
            multiple of the chunk size.
        """
        self.store_kv(tokens, mask, **kwargs)

    def store_kv(self,
                 tokens: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 **kwargs) -> None:

        if mask is not None:
            monitor_req_id = self.stats_monitor.on_store_request(
                torch.sum(mask))
        else:
            monitor_req_id = self.stats_monitor.on_store_request(len(tokens))

        last_token_idx = 0
        for start, end, key in self.token_database.process_tokens(
                tokens, mask):
            if self.storage_manager.contains(key):
                continue
            # Allocate the memory object
            num_tokens = end - start
            kv_shape = self.gpu_connector.get_shape(num_tokens)
            kv_dtype = self.metadata.kv_dtype
            memory_obj = self.storage_manager.allocate(kv_shape, kv_dtype)
            if memory_obj is None:
                logger.warning("Failed to allocate memory for the KV cache.\n"
                               "The KV cache will not be stored.")
                break

            # Put the memory object to the storage backend
            # Disabling put_queue for now, as it's not necessary
            # and bringing big overhead
            # self.put_queue.put((key, memory_obj, start, end, kwargs))

            self.gpu_connector.from_gpu(memory_obj, start, end, **kwargs)
            self.storage_manager.put(key, memory_obj)

            # Update lookup server
            if self.lookup_server is not None:
                self.lookup_server.insert(key)
            last_token_idx = end

        self.stats_monitor.on_store_finished(monitor_req_id)

        if last_token_idx == len(tokens):
            self.store_hidden_states(tokens, kwargs.get("hidden_states"))

    def store_hidden_states(self, tokens: torch.Tensor,
                            hidden_states: torch.Tensor) -> None:

        if hidden_states is None:
            return

        #TODO: support codegen serde mode
        if self.config.remote_serde != "naive":
            logger.warning(
                "Hidden states storage only supports in naive serde mode.")
            return

        hidden_states_key = self.token_database.make_hidden_states_key(tokens)

        if self.storage_manager.contains(hidden_states_key):
            return

        # the LMCache backend assumes a tensor with 4 dimensions
        assert len(hidden_states.shape) == 2
        hidden_states = hidden_states.unsqueeze(0).unsqueeze(0)

        memory_obj = self.storage_manager.allocate(hidden_states.shape,
                                                   hidden_states.dtype)
        if memory_obj is None or memory_obj.tensor is None:
            logger.warning("Failed to allocate memory for the hidden states.")
            return

        memory_obj.tensor.copy_(hidden_states)
        self.storage_manager.put(hidden_states_key, memory_obj)

    def retrieve_hidden_states(self,
                               tokens: torch.Tensor) -> Optional[torch.Tensor]:

        #TODO: support codegen serde mode
        if self.config.remote_serde != "naive":
            logger.warning(
                "Hidden states retrieval is supported in serde mode only.")
            return None

        hidden_states_key = self.token_database.make_hidden_states_key(tokens)
        memory_obj = self.storage_manager.get(hidden_states_key)
        if memory_obj is None or memory_obj.tensor is None:
            logger.warning("Failed to retrieve the hidden states.")
            return None

        # the LMCache backend demands a tensor with 4 dimensions
        # change it back
        assert len(memory_obj.tensor.shape) == 4
        hidden_states = memory_obj.tensor.squeeze(0).squeeze(0)

        return hidden_states

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve(self,
                 tokens: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 **kwargs) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should 
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched, 
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables). 

        :return: the boolean mask indicating which tokens are retrieved. The 
            length of the mask should be the same as the tokens. On CPU.

        :return: the hidden states of the tokens. If all the tokens are
            retrieved, the hidden states will be returned. Otherwise, None.

        :raises: ValueError if the number of Falses in the mask is not a 
            multiple of the chunk size.
        """
        if mask is not None:
            monitor_req_id = self.stats_monitor.on_retrieve_request(
                torch.sum(mask))
        else:
            monitor_req_id = self.stats_monitor.on_retrieve_request(
                len(tokens))

        ret_mask = torch.zeros_like(tokens, dtype=torch.bool, device="cpu")
        # to track how many tokens are retrieved and decide whether the
        # hidden states should be retrieved
        last_token_idx = 0
        for start, end, key in self.token_database.process_tokens(
                tokens, mask):

            # Get the memory object from the storage backend
            memory_obj = self.storage_manager.get(key)

            if memory_obj is None:
                if self.enable_p2p:
                    future_memory_obj = asyncio.run_coroutine_threadsafe(
                        self.distributed_server.issue_get(key),
                        self.distributed_loop)
                    memory_obj = future_memory_obj.result()
                if memory_obj is None:
                    break

            ret_mask[start:end] = True
            last_token_idx = end

            # NOTE(Jiayi): memory_obj doesn't have to be a pinned
            # cpu tensor for the sake of performance.
            # For example, disk->gpu is faster than disk->cpu->gpu.
            # RDMA is another example.

            self.gpu_connector.to_gpu(memory_obj, start, end, **kwargs)
            self.memory_allocator.ref_count_down(memory_obj)

        self.stats_monitor.on_retrieve_finished(monitor_req_id,
                                                torch.sum(ret_mask))

        hidden_states = self.retrieve_hidden_states(tokens) \
            if last_token_idx == len(tokens) \
            else None

        return ret_mask, hidden_states

    def prefetch(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Launch the prefetching process in the storage manager to load the 
        KV to the local CPU memory
        """
        for start, end, key in self.token_database.process_tokens(
                tokens, mask):
            self.storage_manager.prefetch(key)

    # TODO(Jiayi): Currently, search_range is only used for testing.
    def lookup(
        self,
        tokens: torch.Tensor,
        search_range: Optional[List[str]] = None,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param tokens: the input tokens, with shape [seq_len]
        
        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of ["Hot", "LocalDiskBackend"] for now.
        If None, search in all backends.

        :return: An int indicating how many prefix tokens are cached.
        """

        for start, end, key in self.token_database.process_tokens(tokens):
            if not self.storage_manager.contains(key, search_range):
                return start
        return end

    def close(self) -> None:
        """Close the cache engine and free all the resources"""

        if self.enable_p2p:
            self.distributed_server.close()

        self.storage_manager.close()
        logger.info("LMCacheEngine closed.")


class LMCacheEngineBuilder:
    _instances: Dict[str, LMCacheEngine] = {}
    _cfgs: Dict[str, LMCacheEngineConfig] = {}
    _metadatas: Dict[str, LMCacheEngineMetadata] = {}
    _stat_loggers: Dict[str, LMCacheStatsLogger] = {}

    @staticmethod
    def _Create_memory_allocator(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ) -> MemoryAllocatorInterface:
        max_local_cpu_size = config.max_local_cpu_size
        return MixedMemoryAllocator(int(max_local_cpu_size * 1024**3))

    @staticmethod
    def _Create_token_database(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ) -> TokenDatabase:
        return ChunkedTokenDatabase(config, metadata)

    @classmethod
    def get_or_create(
            cls,
            instance_id: str,
            config: LMCacheEngineConfig,
            metadata: LMCacheEngineMetadata,
            gpu_connector:
        GPUConnectorInterface,  # gpu connectors is from outside
    ) -> LMCacheEngine:
        """
        Builds a new LMCacheEngine instance if it doesn't already exist for the
        given ID.

        raises: ValueError if the instance already exists with a different
            configuration.
        """
        logger.info(f"Creating LMCacheEngine instance {instance_id}")
        if instance_id not in cls._instances:
            memory_allocator = cls._Create_memory_allocator(config, metadata)
            token_database = cls._Create_token_database(config, metadata)
            stat_logger = LMCacheStatsLogger(metadata, log_interval=10)
            engine = LMCacheEngine(config, metadata, memory_allocator,
                                   token_database, gpu_connector)
            cls._instances[instance_id] = engine
            cls._cfgs[instance_id] = config
            cls._metadatas[instance_id] = metadata
            cls._stat_loggers[instance_id] = stat_logger
            return engine
        else:
            if (cls._cfgs[instance_id] != config
                    or cls._metadatas[instance_id] != metadata):
                raise ValueError(
                    f"Instance {instance_id} already exists with a different "
                    f"configuration or metadata.")
            return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[LMCacheEngine]:
        """Returns the LMCacheEngine instance associated with the instance ID, 
        or None if not found."""
        return cls._instances.get(instance_id)

    @classmethod
    def destroy(cls, instance_id: str) -> None:
        """Close and delete the LMCacheEngine instance by the instance ID"""
        # TODO: unit test for this
        if instance_id in cls._instances:
            stat_logger = cls._stat_loggers[instance_id]
            stat_logger.shutdown()
            engine = cls._instances[instance_id]
            engine.close()
            cls._instances.pop(instance_id, None)
            cls._cfgs.pop(instance_id, None)
            cls._metadatas.pop(instance_id, None)
            cls._stat_loggers.pop(instance_id, None)
            LMCStatsMonitor.DestroyInstance()
