"""chunk_kda's fused Q/K/V prepare: one launch for strided q/k/v views (column
slices of a packed conv output, as in Kimi-K3 KDA), bit-identical to the
contiguous() + l2norm_fwd path it replaces."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.fla.kda import _fuse_qkv_prepare, chunk_kda
from sglang.kernels.ops.attention.fla.l2norm import (
    l2norm_fwd,
    l2norm_qk_prepare_v_fwd,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")


def _packed_qkv_views(tokens, qk_heads, v_heads, head_dim, dtype, seed=0):
    """[tokens, H, D] q/k/v views into one token-major packed buffer."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    qk_width = qk_heads * head_dim
    v_width = v_heads * head_dim
    packed = torch.randn(
        tokens,
        2 * qk_width + v_width,
        generator=generator,
        device="cuda",
        dtype=dtype,
    )
    q = packed[:, :qk_width].unflatten(-1, (qk_heads, head_dim))
    k = packed[:, qk_width : 2 * qk_width].unflatten(-1, (qk_heads, head_dim))
    v = packed[:, 2 * qk_width :].unflatten(-1, (v_heads, head_dim))
    return q, k, v


def _assert_bit_identical(test, actual, expected):
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertEqual(actual.shape, expected.shape)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@unittest.skipIf(not torch.cuda.is_available(), "Test requires a GPU")
class TestL2NormQkPrepareV(CustomTestCase):
    @torch.inference_mode()
    def test_matches_separate_calls(self):
        for tokens in (1, 2, 15, 16, 17, 33, 1000, 8192):
            for head_dim in (64, 128, 256, 512):
                for dtype in (torch.bfloat16, torch.float16, torch.float32):
                    with self.subTest(tokens=tokens, head_dim=head_dim, dtype=dtype):
                        q, k, v = _packed_qkv_views(tokens, 3, 5, head_dim, dtype)
                        q_out, k_out, v_out = l2norm_qk_prepare_v_fwd(q, k, v)
                        _assert_bit_identical(self, q_out, l2norm_fwd(q.contiguous()))
                        _assert_bit_identical(self, k_out, l2norm_fwd(k.contiguous()))
                        _assert_bit_identical(self, v_out, v.contiguous())
                        self.assertTrue(q_out.is_contiguous())
                        self.assertTrue(k_out.is_contiguous())
                        self.assertTrue(v_out.is_contiguous())

    @torch.inference_mode()
    def test_large_and_small_magnitudes(self):
        for scale in (1e-2, 1.0, 1e2):
            with self.subTest(scale=scale):
                q, k, v = _packed_qkv_views(257, 12, 12, 128, torch.bfloat16)
                q, k = q * scale, k * scale
                q_out, k_out, _ = l2norm_qk_prepare_v_fwd(q, k, v)
                _assert_bit_identical(self, q_out, l2norm_fwd(q.contiguous()))
                _assert_bit_identical(self, k_out, l2norm_fwd(k.contiguous()))

    def test_rejects_unsupported_inputs(self):
        q, k, v = _packed_qkv_views(4, 2, 2, 1024, torch.bfloat16)
        with self.assertRaises(ValueError):
            l2norm_qk_prepare_v_fwd(q, k, v)
        q, k, v = _packed_qkv_views(4, 2, 2, 128, torch.bfloat16)
        with self.assertRaises(ValueError):
            l2norm_qk_prepare_v_fwd(q, k[:, :1], v)
        with self.assertRaises(ValueError):
            l2norm_qk_prepare_v_fwd(q, k, v.float())


def _kda_inputs(tokens, heads, head_dim, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = _packed_qkv_views(tokens, heads, heads, head_dim, torch.bfloat16, seed)
    # In place, so v stays a strided view: chunk_kda writes its output into a
    # contiguous v.
    v.mul_(0.1)
    gate = torch.randn(
        1, tokens, heads, head_dim, generator=generator, device="cuda"
    ).to(torch.bfloat16)
    beta = torch.randn(1, tokens, heads, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    a_log = torch.randn(heads, generator=generator, device="cuda")
    dt_bias = torch.randn(heads * head_dim, generator=generator, device="cuda")
    return q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), gate, beta, a_log, dt_bias


def _run_chunk_kda(q, k, v, gate, beta, a_log, dt_bias, seq_lens):
    """chunk_kda on copies of the non-q/k/v inputs, which it may mutate."""
    gate, beta, a_log, dt_bias = (t.clone() for t in (gate, beta, a_log, dt_bias))
    heads, head_dim = q.shape[2], q.shape[3]
    state = torch.full(
        (len(seq_lens), heads, head_dim, head_dim),
        0.01,
        device="cuda",
        dtype=torch.float32,
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.long,
    )
    out = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        initial_state=state,
        initial_state_indices=torch.arange(
            len(seq_lens), device="cuda", dtype=torch.int32
        ),
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
    )
    return out, state


@unittest.skipIf(not torch.cuda.is_available(), "Test requires a GPU")
class TestChunkKdaFusedPrepare(CustomTestCase):
    @torch.inference_mode()
    def test_strided_views_match_contiguous_inputs(self):
        for seq_lens in ([100], [64, 1, 191], [2048]):
            with self.subTest(seq_lens=seq_lens):
                q, k, v, gate, beta, a_log, dt_bias = _kda_inputs(
                    sum(seq_lens), heads=12, head_dim=128, seed=len(seq_lens)
                )
                self.assertTrue(_fuse_qkv_prepare(q, k, v))
                expected, expected_state = _run_chunk_kda(
                    q.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    gate,
                    beta,
                    a_log,
                    dt_bias,
                    seq_lens,
                )
                v_before = v.clone()
                actual, actual_state = _run_chunk_kda(
                    q, k, v, gate, beta, a_log, dt_bias, seq_lens
                )
                _assert_bit_identical(self, v, v_before)
                _assert_bit_identical(self, actual, expected)
                _assert_bit_identical(self, actual_state, expected_state)

    def test_only_strided_batch_one_inputs_take_the_fused_path(self):
        q, k, v, *_ = _kda_inputs(8, heads=2, head_dim=128, seed=0)
        self.assertTrue(_fuse_qkv_prepare(q, k, v))
        self.assertFalse(
            _fuse_qkv_prepare(q.contiguous(), k.contiguous(), v.contiguous())
        )
        self.assertFalse(_fuse_qkv_prepare(q[:, :0], k[:, :0], v[:, :0]))
        batch_two = q.expand(2, -1, -1, -1)
        self.assertFalse(_fuse_qkv_prepare(batch_two, batch_two, batch_two))
        q_wide, k_wide, v_wide, *_ = _kda_inputs(8, heads=2, head_dim=1024, seed=0)
        self.assertFalse(_fuse_qkv_prepare(q_wide, k_wide, v_wide))

    @torch.inference_mode()
    def test_fused_prepare_launches_no_copy(self):
        q, k, v, gate, beta, a_log, dt_bias = _kda_inputs(
            64, heads=4, head_dim=128, seed=0
        )
        fused_kernels = _profiled_kernel_names(
            lambda: _run_chunk_kda(q, k, v, gate, beta, a_log, dt_bias, [64])
        )
        self.assertTrue(
            any("prefill_qkv_prepare_kernel" in name for name in fused_kernels)
        )
        self.assertFalse(
            any("l2norm_fwd_kernel" in name for name in fused_kernels), fused_kernels
        )
        self.assertFalse(
            any("direct_copy" in name for name in fused_kernels), fused_kernels
        )

        with patch(f"{chunk_kda.__module__}._fuse_qkv_prepare", return_value=False):
            stock_kernels = _profiled_kernel_names(
                lambda: _run_chunk_kda(q, k, v, gate, beta, a_log, dt_bias, [64])
            )
        self.assertTrue(any("direct_copy" in name for name in stock_kernels))


def _profiled_kernel_names(run):
    run()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        run()
        torch.cuda.synchronize()
    return [
        event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]


if __name__ == "__main__":
    unittest.main()
