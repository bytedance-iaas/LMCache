import dataclasses
from copy import deepcopy
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.attention.backends.flashmla import FlashMLAMetadata
from vllm.attention.backends.mla.common import MLACommonMetadata
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, KVTransferConfig
from vllm.sequence import IntermediateTensors
from vllm.utils import align_to_256bytes, get_kv_cache_torch_dtype

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.gpu_connector import (
    GPUConnectorInterface,
    VLLMPagedMemGPUConnectorMLA,
    VLLMPagedMemGPUConnectorV2,
)
from lmcache.integration.vllm.utils import ENGINE_NAME, lmcache_get_config
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

# FIXME(Jiayi): temporarily comment this out
# from lmcache_vllm.blend_adapter import remove_request_id_indices

logger = init_logger(__name__)

LMCACHE_CUDA_STREAM = torch.cuda.Stream()


class StoreStatus(Enum):
    PREFILL = 1
    CHUNK_PREFILL = 2
    DECODE = 3
    SUFFIX_PREFILL = 4
    NONE = 5


class RetrieveStatus(Enum):
    PREFILL = 1  # include (1) normal_prefill
    # (2) chunk_prefill_last
    # (3) prefix_prefill
    CHUNK_PREFILL = 2  # not last chunk
    NONE = 4


SUPPORTED_BACKEND_METADATA = (
    FlashAttentionMetadata,
    FlashMLAMetadata,
    MLACommonMetadata,
)


def init_lmcache_engine(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    cache_config: CacheConfig,
    kv_transfer_config: KVTransferConfig,
) -> Optional[LMCacheEngine]:
    """Initialize the LMCache engine by the given model config and parallel
    config. This function will check the environment variable
    `LMCACHE_CONFIG_FILE` to load the configuration file. If that environment
    variable is not set, this function will return None.

    :param model_config: The model configuration in vLLM.
    :type model_config: ModelConfig
    :param parallel_config: The parallel configuration in vLLM.
    :type parallel_config: ParallelConfig
    :param cache_config: The KV cache configuration in vLLM.
    :type cache_config: CacheConfig

    :return: The initialized LMCache engine or None (if the environment variable
        `LMCACHE_CONFIG_FILE` is not set).
    :rtype: Optional[LMCacheEngine]
    """
    if LMCacheEngineBuilder.get(ENGINE_NAME) is not None:
        return None

    config = lmcache_get_config()

    kv_dtype = get_kv_cache_torch_dtype(cache_config.cache_dtype, model_config.dtype)

    use_mla = False
    if (
        hasattr(model_config, "use_mla")
        and isinstance(model_config.use_mla, bool)
        and model_config.use_mla
    ):
        use_mla = True

    # construct kv shape (for mem pool)
    num_layer = model_config.get_num_layers(parallel_config)
    chunk_size = config.chunk_size
    num_kv_head = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()

    if use_mla:
        kv_shape = (num_layer, 1, chunk_size, 1, head_size)
    else:
        kv_shape = (num_layer, 2, chunk_size, num_kv_head, head_size)
        
    is_kv_producer = kv_transfer_config.is_kv_producer
    is_kv_consumer = kv_transfer_config.is_kv_consumer

    # Change current device.
    torch.cuda.device(parallel_config.rank)
    metadata = LMCacheEngineMetadata(model_config.model,
                                     parallel_config.world_size,
                                     parallel_config.rank, "vllm", kv_dtype,
                                     kv_shape, use_mla, is_kv_producer,
                                     is_kv_consumer)

    vllm_gpu_connector: GPUConnectorInterface
    if use_mla:
        aligned_head_size = align_to_256bytes(head_size, kv_dtype)
        vllm_gpu_connector = VLLMPagedMemGPUConnectorMLA(aligned_head_size, num_layer)
    else:
        hidden_dim_size = num_kv_head * head_size
        vllm_gpu_connector = VLLMPagedMemGPUConnectorV2(
            hidden_dim_size, num_layer)

    assert isinstance(config, LMCacheEngineConfig), \
        "LMCache experimental configuration is should be passed."
    engine = LMCacheEngineBuilder.get_or_create(ENGINE_NAME, config, metadata,
                                                vllm_gpu_connector)

    return engine


def broadcast_seq_group_list(
    model_input: "ModelInputForGPUWithSamplingMetadata",
    parallel_config: ParallelConfig,
    is_driver_worker: bool,
) -> "ModelInputForGPUWithSamplingMetadata":
    """Broadcast the `model_input` from driver worker to non-driver workers.

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param is_driver_worker: Whether the code is executed in driver worker.
    :type is_driver_worker: bool

    : return: Original `model_input` if driver_worker.
              Broadcasted `model_input` otherwise.
    """
    # No need to broadcast if there is only one worker
    if parallel_config.world_size <= 1:
        return model_input

    # broadcast len of `seq_group_metadata_list`
    if is_driver_worker:
        assert model_input.sampling_metadata is not None
        assert model_input.sampling_metadata.seq_groups is not None
        seq_group_len_list = [len(model_input.sampling_metadata.seq_groups)]
    else:
        seq_group_len_list = [0]
    dist.broadcast_object_list(seq_group_len_list, src=0)
    seq_group_len = seq_group_len_list[0]

    # broadcast `seq_groups`
    if is_driver_worker:
        seq_groups = model_input.sampling_metadata.seq_groups  # type: ignore
    else:
        seq_groups = [None] * seq_group_len
    dist.broadcast_object_list(seq_groups, src=0)

    if is_driver_worker:
        return model_input
    else:
        sampling_metadata = model_input.sampling_metadata
        sampling_metadata.seq_groups = seq_groups  # type: ignore
        return dataclasses.replace(model_input, sampling_metadata=sampling_metadata)


def close_lmcache_engine() -> None:
    """Close the LMCache engine if it is initialized."""
    logger.debug("Closing LMCache Engine")
    LMCacheEngineBuilder.destroy(ENGINE_NAME)


# This function is not used for now
def lmcache_should_retrieve(
    model_input: "ModelInputForGPUWithSamplingMetadata",
) -> List[RetrieveStatus]:
    """Check should we retrieve KV from LMCache for the current model_input.

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param kv_caches: The paged memory
    :type kv_caches: List[torch.Tensor]

    :return: RetrieveStatus.
    """
    # model_input doesn't have seq_lens in tp
    # but attn_metadata does
    seq_lens = model_input.attn_metadata.seq_lens
    assert seq_lens is not None
    num_seqs = len(seq_lens)
    retrieve_status = [RetrieveStatus.NONE] * num_seqs

    attn_meta = model_input.attn_metadata

    prefill_exist = attn_meta.num_prefills > 0
    if not prefill_exist:
        return retrieve_status
    assert model_input.sampling_metadata is not None
    seq_group_list = model_input.sampling_metadata.seq_groups
    assert seq_group_list is not None

    seq_data_idx = 0
    # selected_token_indices_idx = 0
    for seq_group_idx, seq_group in enumerate(seq_group_list):
        num_seqs_in_seq_group = len(seq_group.seq_data)
        seq_data_idx_end = seq_data_idx + num_seqs_in_seq_group

        # DECODE
        if not seq_group.is_prompt:
            seq_data_idx = seq_data_idx_end
            continue

        # CHUNK_PREFILL
        if not seq_group.do_sample:
            retrieve_status[seq_data_idx:seq_data_idx_end] = [
                RetrieveStatus.CHUNK_PREFILL
            ] * num_seqs_in_seq_group
            seq_data_idx = seq_data_idx_end
        # LAST_CHUNK_PREFILL or NORMAL_PREFILL
        else:
            retrieve_status[seq_data_idx:seq_data_idx_end] = [
                RetrieveStatus.PREFILL
            ] * num_seqs_in_seq_group
            seq_data_idx = seq_data_idx_end

    return retrieve_status


def lmcache_should_store(
    model_input: "ModelInputForGPUWithSamplingMetadata",
    engine: LMCacheEngine,
) -> List[StoreStatus]:
    """Check should we store KV into LMCache for the current model_input.

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata


    :return: A list of StoreStatus.
             StoreStatus.PREFILL/DECODE/CHUNK_PREFILL if
             we should store KV after PREFILL/DECODE.
             StoreStatus.NONE if no storing is required.
    """

    def is_blend_effective(attn_metadata):
        """Check if the blend is effective for the current request"""
        blend_metadata = getattr(attn_metadata, "blend_metadata", None)
        if blend_metadata is None:
            return False

        return blend_metadata.processed_layer_count > 0

    seq_lens = model_input.attn_metadata.seq_lens
    assert seq_lens is not None
    store_status = [StoreStatus.NONE] * len(seq_lens)

    attn_meta = model_input.attn_metadata

    # Don't store if this request is processed by cacheblend
    if is_blend_effective(attn_meta):
        return store_status

    assert model_input.sampling_metadata is not None

    seq_group_list = model_input.sampling_metadata.seq_groups
    assert seq_group_list is not None

    selected_token_indices = model_input.sampling_metadata.selected_token_indices

    seq_data_idx = 0
    selected_token_indices_idx = 0
    for seq_group_idx, seq_group in enumerate(seq_group_list):
        num_seqs_in_seq_group = len(seq_group.seq_data)
        seq_data_idx_end = seq_data_idx + num_seqs_in_seq_group

        # DECODE
        if not seq_group.is_prompt:
            # Determine whether to save decoded KV cache
            if not engine.config.save_decode_cache:
                for idx in range(seq_data_idx, seq_data_idx_end):
                    if seq_lens[idx] % engine.config.chunk_size == 0:
                        store_status[idx] = StoreStatus.DECODE
            seq_data_idx = seq_data_idx_end
            selected_token_indices_idx += num_seqs_in_seq_group
            continue

        # TODO(Jiayi): Maybe it's cleaner to handle all logic for
        # `lmcache_model_request` inside `cache_engine`
        # Check whether user has specified to not store the cache
        if hasattr(seq_group, "lmcache_model_request"):
            lmcache_model_request = seq_group.lmcache_model_request
            if lmcache_model_request is not None:
                user_should_store = lmcache_model_request.store_cache
                if not user_should_store:
                    logger.debug("User has specified not to store the cache")
                    seq_data_idx += len(seq_group.seq_data)
                    continue

        # CHUNK_PREFILL
        if not seq_group.do_sample:
            store_status[seq_data_idx:seq_data_idx_end] = [
                StoreStatus.CHUNK_PREFILL
            ] * num_seqs_in_seq_group
            seq_data_idx = seq_data_idx_end
            continue

        # LAST_CHUNK_PREFILL or NORMAL_PREFILL
        for seqid, seq_data in seq_group.seq_data.items():
            if (
                seq_data.get_len() - 1
                != selected_token_indices[selected_token_indices_idx]
            ):
                # last chunk in chunk prefill
                # or prefix already hit in retrieve
                store_status[seq_data_idx] = StoreStatus.SUFFIX_PREFILL
            else:
                store_status[seq_data_idx] = StoreStatus.PREFILL
            seq_data_idx += 1
            selected_token_indices_idx += 1
    return store_status


@_lmcache_nvtx_annotate
def lmcache_store_kv(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    cache_config: CacheConfig,
    model_executable: torch.nn.Module,
    model_input: "ModelInputForGPUWithSamplingMetadata",
    kv_caches: List[torch.Tensor],
    hidden_states: Optional[torch.Tensor] = None,
) -> None:
    """Store the KV caches into LMCache for the current model_input.

    :param model_executable: The model executable for the current request.
    :type model_executable: torch.nn.Module

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param kv_caches: The paged memory to get KV from
    :type kv_caches: List[torch.Tensor]

    :param store_status: Indicate whether and how KV cache of each req is stored
    :type store_status: List[StoreStatus]
    """
    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    assert engine is not None, "LMCache engine is not initialized."

    assert isinstance(
        model_input.attn_metadata, SUPPORTED_BACKEND_METADATA
    ), f"Only backend with {SUPPORTED_BACKEND_METADATA} is supported for now."

    seq_lens = model_input.attn_metadata.seq_lens
    assert seq_lens is not None

    slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
    assert slot_mapping is not None

    # query_start_loc = model_input.attn_metadata.query_start_loc
    # assert query_start_loc is not None

    request_ids = model_input.request_ids
    assert request_ids is not None

    # TODO (Jiayi): commenting the following out for now
    # as Turing architecture is not supported yet
    # For Turing GPU
    # num_heads = model_config.get_num_kv_heads(parallel_config)
    # head_size = model_config.get_head_size()
    # gpu_capability = torch.cuda.get_device_capability()

    # assert model_input.sampling_metadata is not None

    # seq_group_list = model_input.sampling_metadata.seq_groups
    # model_input = broadcast_seq_group_list(model_input, parallel_config,
    #                                        seq_group_list is not None)
    # seq_group_list = model_input.sampling_metadata.seq_groups
    # assert seq_group_list is not None
    input_tokens = model_input.input_tokens

    # store_status = lmcache_should_store(model_input, engine)
    # store_status = [StoreStatus.PREFILL] * len(seq_lens)

    next_start_pos = 0
    for seq_data_idx, slen in enumerate(seq_lens):
        start_pos = next_start_pos
        end_pos = start_pos + slen
        seq_len = seq_lens[seq_data_idx]

        current_tokens = input_tokens[start_pos:end_pos]
        slot_mapping_req_full = slot_mapping[start_pos:end_pos]
        skip_leading_tokens = 0
        kv_tensors_mask = None
        token_mask = torch.ones_like(current_tokens, dtype=torch.bool)

        engine.store(
            current_tokens.cpu(),
            token_mask,
            kvcaches=kv_caches,
            slot_mapping=slot_mapping_req_full,
            offset=skip_leading_tokens,
            request_id = request_ids[seq_data_idx],
            hidden_states=hidden_states,
        )
        seq_data_idx += 1


@_lmcache_nvtx_annotate
def lmcache_retrieve_kv(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    cache_config: CacheConfig,
    model_executable: torch.nn.Module,
    model_input: "ModelInputForGPUWithSamplingMetadata",
    kv_caches: List[torch.Tensor],
) -> Tuple[
    "ModelInputForGPUWithSamplingMetadata",
    bool,
    Union[torch.Tensor, IntermediateTensors],
]:
    """Retrieve the KV caches from LMCache for the current model_input. And
    rebuild the model_input to reflect the changes in KV if necessary.

    :param model_executable: The model executable for the current request.
    :type model_executable: torch.nn.Module

    :param model_input: The model input for the current request.
    :type model_input: ModelInputForGPUWithSamplingMetadata

    :param kv_caches: The paged memory to put KV to
    :type kv_caches: List[torch.Tensor]

    :param retrieve_status: Indicate whether and how
                            KV cache of each req is retrieved
    :type retrieve_status: List[RetrieveStatus]

    :return: The rebuilt model_input to reflect the changes in KV.
    :return: The boolean value to indicate whether the
             entire execute_model should be skipped
    """

    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    assert engine is not None, "LMCache engine is not initialized."

    if engine.config.enable_blending:
        return model_input, False, None

    assert isinstance(
        model_input.attn_metadata, SUPPORTED_BACKEND_METADATA
    ), f"Only backend with {SUPPORTED_BACKEND_METADATA} is supported for now."

    query_start_loc = model_input.attn_metadata.query_start_loc
    assert query_start_loc is not None
    slot_mapping = model_input.attn_metadata.slot_mapping.flatten()

    assert slot_mapping is not None
    seq_lens = model_input.attn_metadata.seq_lens
    assert seq_lens is not None

    input_tokens = model_input.input_tokens

    # The following metadata are needed to rebuilt the model input
    full_tokens_list = []
    num_computed_tokens_list = []
    lmc_num_computed_tokens_list = []

    start_pos_list = []
    is_prefill_list = []

    do_sample_list = []

    next_start_pos = 0
    num_request_not_found = 0

    # idx is on a sequence, not a sequence group.
    idx = 0

    # assert model_input.sampling_metadata is not None
    # seq_group_list = model_input.sampling_metadata.seq_groups
    # model_input = broadcast_seq_group_list(model_input, parallel_config,
    #                                        seq_group_list is not None)
    # seq_group_list = model_input.sampling_metadata.seq_groups
    # assert seq_group_list is not None

    hidden_states_list = []

    # TODO: hardcode the retrieve status in vllm_adapter to avoid
    # index error, need to fix this.
    retrieve_status = [RetrieveStatus.PREFILL] * len(seq_lens)

    for seq_data_idx, slen in enumerate(seq_lens):
        start_pos = next_start_pos
        end_pos = start_pos + slen
        total_seq_len = seq_lens[seq_data_idx]

        full_token_tensor = input_tokens[start_pos:end_pos]
        full_tokens_list.append(full_token_tensor)

        vllm_num_required_tokens = end_pos - start_pos
        assert isinstance(vllm_num_required_tokens, int)

        next_start_pos = end_pos
        start_pos_list.append(start_pos)

        # number of tokens already computed by vllm
        # (e.g., chunk prefill, prefix caching)
        vllm_num_computed_tokens = total_seq_len - vllm_num_required_tokens

        # NOTE: No need to retrieve from lmc if the number of tokens
        # to be retrieved is small
        lmc_chunk_size = engine.config.chunk_size
        if vllm_num_required_tokens < lmc_chunk_size:
            num_computed_tokens_list.append(vllm_num_computed_tokens)
            lmc_num_computed_tokens_list.append(0)
            idx += 1
            num_request_not_found += 1
            continue

        # construct token mesk to indicate what tokens should be retrieved
        # from lmc. Tokens computed in vllm already should be skipped
        token_mask = torch.ones_like(full_token_tensor, dtype=torch.bool)
        vllm_num_computed_tokens_align = (
            vllm_num_computed_tokens // lmc_chunk_size * lmc_chunk_size
        )
        token_mask[:vllm_num_computed_tokens_align] = False

        slot_mapping_req_full = slot_mapping[start_pos:end_pos]

        # call lmcache retrieve
        ret_token_mask, seq_hidden_states = engine.retrieve(
            full_token_tensor.cpu(),
            token_mask,
            kvcaches=kv_caches,
            slot_mapping=slot_mapping_req_full,
            use_mla=engine.metadata.use_mla,
        )
        lmc_num_computed_tokens = max(
            torch.sum(ret_token_mask).item()
            - (vllm_num_computed_tokens - vllm_num_computed_tokens_align),
            0,
        )

        assert isinstance(lmc_num_computed_tokens, int)

        # total number of computed tokens (vllm + lmc)
        num_computed_tokens = vllm_num_computed_tokens + lmc_num_computed_tokens

        # Avoid error when prefix is exactly the same as the retrieved
        # However, the entire prefill should be skipped in chunk prefill
        if num_computed_tokens == total_seq_len:
            lmc_num_computed_tokens -= 1
            num_computed_tokens -= 1

        num_computed_tokens_list.append(num_computed_tokens)
        lmc_num_computed_tokens_list.append(lmc_num_computed_tokens)

        # No cache found, move on
        if lmc_num_computed_tokens == 0 or seq_hidden_states is None:
            num_request_not_found += 1

        hidden_states_list.append(seq_hidden_states)

        # Inject the lmc retrieved kv cache
        logger.debug(f"Injected token number: {lmc_num_computed_tokens}")

        idx += 1

    seq_cnt = len(query_start_loc) - 1
    assert idx == seq_cnt
    assert len(lmc_num_computed_tokens_list) == seq_cnt
    assert len(num_computed_tokens_list) == seq_cnt

    is_all_prefill = all(
        [status == RetrieveStatus.PREFILL for status in retrieve_status]
    )

    if is_all_prefill and num_request_not_found == 0:
        device = kv_caches[0].device
        hidden_states = torch.cat(hidden_states_list, dim=0).to(device)
        return model_input, True, hidden_states


    logger.debug("Returning the original input!")
    return model_input, False, None
