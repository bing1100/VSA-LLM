import pytest
import torch

from vsa_embed.context import CausalLocalContext, FrameSidecar


@pytest.mark.parametrize("mode", ["mean", "conv"])
def test_local_context_is_causal(mode: str) -> None:
    torch.manual_seed(0)
    encoder = CausalLocalContext(8, 4, window=3, mode=mode)
    x = torch.randn(2, 10, 8)
    changed = x.clone(); changed[:, 6:] = torch.randn(2, 4, 8)
    torch.testing.assert_close(encoder(x)[:, :6], encoder(changed)[:, :6])
    assert not torch.allclose(encoder(x)[:, 6:], encoder(changed)[:, 6:])


def test_mean_mode_averages_the_window() -> None:
    encoder = CausalLocalContext(2, 2, window=3, mode="mean")
    with torch.no_grad():
        encoder.projection.weight.copy_(torch.eye(2))
    x = torch.arange(10.0).reshape(1, 5, 2)
    out = encoder(x)
    torch.testing.assert_close(out[0, 0], x[0, 0])
    torch.testing.assert_close(out[0, 4], x[0, 2:5].mean(0))


def test_padding_is_ignored_in_the_mean() -> None:
    encoder = CausalLocalContext(2, 2, window=4, mode="mean")
    x = torch.randn(1, 3, 2)
    mask = torch.tensor([[0, 1, 1]])
    out_masked = encoder(x, mask)
    expected = encoder(x[:, 1:])[:, -1]
    torch.testing.assert_close(out_masked[:, -1], expected)


def test_sidecar_starts_as_identity_and_attention_sums_to_one() -> None:
    sidecar = FrameSidecar(8, 6, key_dimension=4)
    hidden, edges = torch.randn(2, 8), torch.randn(5, 6)
    segments = torch.tensor([0, 0, 1, 1, 1])
    injection, weights = sidecar(hidden, edges, segments)
    torch.testing.assert_close(injection, torch.zeros_like(injection))
    totals = torch.zeros(2).index_add(0, segments, weights)
    torch.testing.assert_close(totals, torch.ones(2))
