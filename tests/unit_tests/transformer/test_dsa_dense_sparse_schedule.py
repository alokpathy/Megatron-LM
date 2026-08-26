# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the DSA dense-to-sparse training schedule (--dsa-indexer-dense-loss-steps)."""

import types

import pytest
import torch

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.training import (
    _clear_dsa_indexer_optimizer_state,
    _dsa_dense_phase_active,
    _reset_dsa_indexer_optimizer_group_steps,
    apply_dsa_dense_sparse_schedule,
)

SPARSE_PHASE_CONFIG = dict(
    num_layers=2,
    hidden_size=32,
    num_attention_heads=4,
    num_query_groups=1,
    kv_channels=8,
    experimental_attention_variant="dsa",
    dsa_indexer_mode="simplified",
    add_bias_linear=False,
    dsa_indexer_topk=4,
    dsa_min_memory_backend="triton-min-memory",
    dsa_indexer_loss_coeff=0.1,
    dsa_indexer_use_sparse_loss=True,
)


def make_config(**overrides):
    """Build a DSA-over-GQA config describing the sparse phase."""
    kwargs = dict(SPARSE_PHASE_CONFIG)
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class TestScheduleConfig:
    """The schedule captures the configured sparse phase and starts the run dense."""

    def test_installs_dense_phase_and_captures_sparse_overrides(self):
        # dsa_kernel_cache_routing is the interesting case: legal in the sparse phase, but
        # rejected outright by the dense-phase assertions because dense bypasses routing.
        config = make_config(dsa_indexer_dense_loss_steps=100, dsa_kernel_cache_routing=True)

        # Live config is the dense phase: dense forward, dense KL target, no caching.
        assert config.dsa_fwd_use_dense_attn
        assert not config.dsa_indexer_use_sparse_loss
        assert not config.dsa_kernel_cache_routing

        # The configured (sparse) values are preserved for the boundary.
        assert config.dsa_sparse_phase_overrides == {
            "dsa_fwd_use_dense_attn": False,
            "dsa_indexer_use_sparse_loss": True,
            "dsa_indexer_sparse_loss_use_topk_only": False,
            "dsa_kernel_cache_routing": True,
            "dsa_kernel_cache_indexer_k": False,
            "dsa_kernel_cache_selected_scores": False,
        }

    def test_no_schedule_leaves_config_untouched(self):
        config = make_config()
        assert config.dsa_sparse_phase_overrides is None
        assert config.dsa_indexer_use_sparse_loss
        assert not config.dsa_fwd_use_dense_attn

    @pytest.mark.parametrize(
        "overrides,message",
        [
            ({"dsa_indexer_dense_loss_steps": 0}, "positive iteration count"),
            (
                {"dsa_indexer_dense_loss_steps": 10, "dsa_min_memory_backend": "reference"},
                "requires a min-memory",
            ),
            (
                {"dsa_indexer_dense_loss_steps": 10, "dsa_train_indexer_only": True},
                "replaces dsa_train_indexer_only",
            ),
            (
                {"dsa_indexer_dense_loss_steps": 10, "dsa_fwd_use_dense_attn": True},
                "leave dsa_fwd_use_dense_attn unset",
            ),
            (
                {"dsa_indexer_dense_loss_steps": 10, "dsa_indexer_use_sparse_loss": False},
                "set dsa_indexer_use_sparse_loss",
            ),
        ],
    )
    def test_rejects_conflicting_settings(self, overrides, message):
        with pytest.raises(AssertionError, match=message):
            make_config(**overrides)

    def test_freezing_is_rejected_because_it_cannot_be_undone(self):
        """dsa_train_indexer_only would make the sparse phase a silent no-op."""
        with pytest.raises(AssertionError, match="lr=0 during the dense phase"):
            make_config(dsa_indexer_dense_loss_steps=10, dsa_train_indexer_only=True)


class FakeOptimizer:
    """Minimal stand-in exposing what the schedule touches on a real MegatronOptimizer."""

    def __init__(self, backbone, indexer, step=0):
        self.optimizer = types.SimpleNamespace(
            state={
                backbone: {
                    "exp_avg": torch.ones_like(backbone),
                    "exp_avg_sq": torch.ones_like(backbone),
                },
                indexer: {
                    "exp_avg": torch.ones_like(indexer),
                    "exp_avg_sq": torch.ones_like(indexer),
                },
            },
            param_groups=[
                {"params": [backbone], "lr": 0.1, "is_dsa_indexer": False, "step": step},
                {"params": [indexer], "lr": 0.1, "is_dsa_indexer": True, "step": step},
            ],
        )
        self._params = [backbone, indexer]

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def get_parameters(self):
        return self._params


def make_model_and_optimizer(dense_steps=100, step=7):
    backbone = torch.nn.Parameter(torch.zeros(4))
    indexer = torch.nn.Parameter(torch.zeros(4))
    config = make_config(dsa_indexer_dense_loss_steps=dense_steps)
    model_chunk = types.SimpleNamespace(config=config)
    model_chunk.named_parameters = lambda: [
        ("decoder.layers.0.self_attention.core_attention.weight", backbone),
        ("decoder.layers.0.self_attention.indexer.linear_q.weight", indexer),
    ]
    return [model_chunk], FakeOptimizer(backbone, indexer, step=step), backbone, indexer


class TestSelectorHelpers:
    """The clear/reset helpers select the backbone as the complement of the indexer."""

    def test_clear_selects_requested_half(self):
        model, optimizer, backbone, indexer = make_model_and_optimizer()

        assert _clear_dsa_indexer_optimizer_state(model, optimizer, indexer=False) == 1
        assert backbone not in optimizer.optimizer.state
        assert indexer in optimizer.optimizer.state

    def test_reset_group_steps_selects_requested_half(self):
        model, optimizer, _, _ = make_model_and_optimizer(step=7)

        assert _reset_dsa_indexer_optimizer_group_steps(optimizer, indexer=False) == 1
        backbone_group, indexer_group = optimizer.param_groups
        assert backbone_group["step"] == 0
        assert indexer_group["step"] == 7, "indexer clock must not be disturbed"


class TestScheduleDriver:
    """Phase transitions as driven from the training loop."""

    def _args(self, dense_steps=100):
        return types.SimpleNamespace(dsa_indexer_dense_loss_steps=dense_steps)

    def test_dense_phase_predicate(self):
        args = self._args(dense_steps=100)
        assert _dsa_dense_phase_active(args, 0)
        assert _dsa_dense_phase_active(args, 99)
        assert not _dsa_dense_phase_active(args, 100)
        assert not _dsa_dense_phase_active(types.SimpleNamespace(), 0)

    def test_dense_phase_masks_backbone_lr_only(self):
        model, optimizer, _, _ = make_model_and_optimizer()
        args = self._args()

        apply_dsa_dense_sparse_schedule(args, model, optimizer, iteration=5)

        backbone_group, indexer_group = optimizer.param_groups
        assert backbone_group["lr"] == 0.0
        assert indexer_group["lr"] == 0.1
        assert args._dsa_schedule_phase == "dense"
        # Config stays in the dense phase.
        assert model[0].config.dsa_fwd_use_dense_attn
        assert not model[0].config.dsa_indexer_use_sparse_loss

    def test_mask_is_reapplied_each_iteration(self):
        """The global scheduler rewrites lr every step, so masking cannot be one-shot."""
        model, optimizer, _, _ = make_model_and_optimizer()
        args = self._args()

        apply_dsa_dense_sparse_schedule(args, model, optimizer, iteration=5)
        optimizer.param_groups[0]["lr"] = 0.1  # scheduler rewrites it
        apply_dsa_dense_sparse_schedule(args, model, optimizer, iteration=6)

        assert optimizer.param_groups[0]["lr"] == 0.0

    def test_boundary_restores_sparse_config_and_clears_backbone_state(self):
        model, optimizer, backbone, indexer = make_model_and_optimizer(dense_steps=10, step=10)
        args = self._args(dense_steps=10)

        apply_dsa_dense_sparse_schedule(args, model, optimizer, iteration=9)
        apply_dsa_dense_sparse_schedule(args, model, optimizer, iteration=10)

        config = model[0].config
        assert not config.dsa_fwd_use_dense_attn
        assert config.dsa_indexer_use_sparse_loss
        assert args._dsa_schedule_phase == "sparse"
        # Both halves of the freeze equivalence: moments dropped and clock reset.
        assert backbone not in optimizer.optimizer.state
        assert optimizer.param_groups[0]["step"] == 0
        # The indexer carries its history across the boundary untouched.
        assert indexer in optimizer.optimizer.state
        assert optimizer.param_groups[1]["step"] == 10

    def test_transition_runs_once(self):
        model, optimizer, backbone, _ = make_model_and_optimizer(dense_steps=10)
        args = self._args(dense_steps=10)

        apply_dsa_dense_sparse_schedule(args, model, optimizer, iteration=9)
        apply_dsa_dense_sparse_schedule(args, model, optimizer, iteration=10)
        optimizer.optimizer.state[backbone] = {"exp_avg": torch.ones(4)}
        optimizer.param_groups[0]["step"] = 3
        apply_dsa_dense_sparse_schedule(args, model, optimizer, iteration=11)

        # Sparse-phase history accumulated after the boundary must survive.
        assert backbone in optimizer.optimizer.state
        assert optimizer.param_groups[0]["step"] == 3

    def test_resume_past_boundary_preserves_optimizer_state(self):
        """A run resuming past the boundary already cleared state before checkpointing."""
        model, optimizer, backbone, _ = make_model_and_optimizer(dense_steps=10, step=42)
        args = self._args(dense_steps=10)

        apply_dsa_dense_sparse_schedule(args, model, optimizer, iteration=500)

        assert not model[0].config.dsa_fwd_use_dense_attn
        assert model[0].config.dsa_indexer_use_sparse_loss
        assert backbone in optimizer.optimizer.state, "resume must not discard sparse history"
        assert optimizer.param_groups[0]["step"] == 42
