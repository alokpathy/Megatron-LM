# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig


def _make_overlap_config(mtp_num_layers: int | None) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=True,
        bf16=True,
        mtp_num_layers=mtp_num_layers,
    )


@pytest.mark.parametrize("mtp_num_layers", [None, 0, 1])
def test_ep_a2a_overlap_accepts_supported_mtp_layer_counts(mtp_num_layers: int | None):
    config = _make_overlap_config(mtp_num_layers)

    assert config.mtp_num_layers == mtp_num_layers


@pytest.mark.parametrize("mtp_num_layers", [-1, 2])
def test_ep_a2a_overlap_rejects_unsupported_mtp_layer_counts(mtp_num_layers: int):
    with pytest.raises(AssertionError, match="MTP supports at most one layer"):
        _make_overlap_config(mtp_num_layers)


def test_batch_invariant_backend_rejects_unknown_value_at_construction():
    # Programmatic construction bypasses argparse's Literal choices, so
    # __post_init__ must catch typos before model init.
    with pytest.raises(AssertionError, match="Unknown batch_invariant_backend"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            batch_invariant_mode=True,
            batch_invariant_backend="te-native",
        )


def test_gdp_num_householder_defaults_to_three():
    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4)

    assert config.gdp_num_householder == 3


def test_gdp_num_householder_accepts_positive_values():
    config = TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=4, gdp_num_householder=5
    )

    assert config.gdp_num_householder == 5


@pytest.mark.parametrize("num_householder", [0, -1])
def test_gdp_num_householder_rejects_non_positive_values(num_householder: int):
    with pytest.raises(ValueError, match="gdp_num_householder must be positive"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            gdp_num_householder=num_householder,
        )


def _make_mxfp8_wire_config(**overrides) -> TransformerConfig:
    kwargs = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="ncclep",
        moe_grouped_gemm=True,
        use_transformer_engine_op_fuser=True,
        moe_dispatch_fwd_dtype='mxfp8',
        moe_combine_bwd_dtype='mxfp8',
        bf16=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_mxfp8_wire_dtypes_accept_valid_ncclep_config():
    config = _make_mxfp8_wire_config()

    assert config.moe_dispatch_fwd_dtype == 'mxfp8'
    assert config.moe_combine_bwd_dtype == 'mxfp8'


def test_mxfp8_wire_dtypes_accept_a2a_overlap():
    # The 1F1B a2a overlap schedule only moves/stages the dispatch output as an opaque block,
    # which the plain-tensor MXFP8 carrier survives; the combination is deliberately allowed.
    config = _make_mxfp8_wire_config(overlap_moe_expert_parallel_comm=True)

    assert config.overlap_moe_expert_parallel_comm


@pytest.mark.parametrize(
    "overrides",
    [
        dict(moe_flex_dispatcher_backend="hybridep"),
        dict(moe_token_dispatcher_type="alltoall", moe_flex_dispatcher_backend=None),
    ],
)
def test_mxfp8_wire_dtypes_reject_non_ncclep_dispatcher(overrides):
    with pytest.raises(ValueError, match="require the 'ncclep' flex"):
        _make_mxfp8_wire_config(**overrides)


@pytest.mark.parametrize(
    "overrides", [dict(use_transformer_engine_op_fuser=False), dict(moe_grouped_gemm=False)]
)
def test_mxfp8_wire_dtypes_require_op_fuser_grouped_gemm(overrides):
    with pytest.raises(ValueError, match="require BOTH"):
        _make_mxfp8_wire_config(**overrides)


def _make_cp_layout_config(**overrides) -> TransformerConfig:
    kwargs = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        context_parallel_size=2,
        attention_cp_layout="contiguous",
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _make_dsa_cp_layout_config(**overrides) -> TransformerConfig:
    # The rest of the DSA validation block is unrelated to CP layouts but has to be satisfied
    # for the config to construct at all: DSA asserts RMSNorm, unfused RoPE, no linear bias, and
    # the simplified indexer currently asserts a single query group.
    kwargs = dict(
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_indexer_topk=16,
        cp_comm_type="allgather",
        normalization="RMSNorm",
        add_bias_linear=False,
        apply_rope_fusion=False,
        num_query_groups=1,
    )
    kwargs.update(overrides)
    return _make_cp_layout_config(**kwargs)


def test_contiguous_attention_cp_layout_rejected_for_ring_attention():
    # Ring and striped attention read the zigzag layout out of raw offsets, so a contiguous
    # shard would be masked as if it sat at the front of the sequence.
    with pytest.raises(ValueError, match="attention_cp_layout='contiguous'"):
        _make_cp_layout_config()


def test_contiguous_attention_cp_layout_allowed_for_dsa_over_gqa():
    # DSA over GQA all-gathers K and V, so the layout only decides which global positions this
    # rank's queries carry -- both layouts describe the same attention.
    config = _make_dsa_cp_layout_config()

    assert config.attention_cp_layout == "contiguous"


def test_contiguous_attention_cp_layout_rejected_for_dsa_over_mla():
    # DSA over MLA runs the upstream MLA CP path, not the GQA all-gather.
    with pytest.raises(ValueError, match="attention_cp_layout='contiguous'"):
        _make_dsa_cp_layout_config(multi_latent_attention=True)


def test_contiguous_attention_cp_layout_is_unrestricted_without_context_parallelism():
    config = _make_cp_layout_config(context_parallel_size=1)

    assert config.attention_cp_layout == "contiguous"
