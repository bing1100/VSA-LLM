import pytest
import torch

transformers = pytest.importorskip("transformers")

from vsa_embed.compose import FrameComposer, FrameSchedule
from vsa_embed.integrations.transformers import ChannelLM, add_lora, chunked_causal_lm_loss
from vsa_embed.span_channel import SpanChannel


def tiny_gpt2() -> torch.nn.Module:
    config = transformers.GPT2Config(vocab_size=97, n_positions=64, n_embd=32, n_layer=2, n_head=2)
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(config).eval()


def tiny_llama() -> torch.nn.Module:
    config = transformers.LlamaConfig(vocab_size=101, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                      tie_word_embeddings=True)
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(config).eval()


@pytest.mark.parametrize("factory", [tiny_gpt2, tiny_llama])
def test_hooked_forward_equals_unhooked_without_spans(factory) -> None:
    model = factory()
    ids = torch.randint(0, 90, (2, 12))
    reference = model(input_ids=ids, labels=ids).loss
    wrapped = ChannelLM(model, channel=None)
    torch.testing.assert_close(wrapped(ids, labels=ids)["loss"], reference, atol=1e-5, rtol=1e-5)


def test_chunked_loss_equals_full_loss_and_per_token_shape() -> None:
    hidden, weight = torch.randn(2, 9, 8), torch.randn(50, 8)
    labels = torch.randint(0, 50, (2, 9)); labels[0, 3] = -100
    logits = hidden @ weight.T
    full = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 50), labels[:, 1:].reshape(-1), ignore_index=-100)
    torch.testing.assert_close(chunked_causal_lm_loss(hidden, weight, labels, chunk=5), full)
    assert chunked_causal_lm_loss(hidden, weight, labels, reduction="none").shape == (2, 8)


def test_frozen_and_lora_hosts_train_only_their_parameters() -> None:
    frozen = ChannelLM(tiny_llama(), host_mode="frozen")
    assert frozen.trainable_parameters() == []
    lora = ChannelLM(tiny_llama(), host_mode="lora", lora_rank=4)
    ids = torch.randint(0, 90, (1, 10))
    before = tiny_llama()(input_ids=ids, labels=ids).loss
    torch.testing.assert_close(lora(ids, labels=ids)["loss"], before, atol=1e-5, rtol=1e-5)  # B = 0 at init
    names = {n for n, p in lora.named_parameters() if p.requires_grad}
    assert names and all("lora_" in n for n in names)


def test_channel_changes_only_linked_positions_through_the_host() -> None:
    model = tiny_gpt2()
    schedule = FrameSchedule.from_frames([[(0, 0), (1, 1)], [(0, 2)]])
    channel = SpanChannel(FrameComposer(schedule, 3, 2, 16), 32, entry_count=2, gate_bias=0.0)
    wrapped = ChannelLM(model, channel)
    ids = torch.randint(0, 90, (1, 10))
    spans = {"batch": torch.tensor([0]), "start": torch.tensor([4]), "end": torch.tensor([5]),
             "inject": torch.tensor([5]), "entry": torch.tensor([0]), "confidence": torch.tensor([1.0]),
             "length": torch.tensor([2])}
    with_channel = wrapped.hidden_states(ids, spans=spans)
    without = wrapped.hidden_states(ids)
    torch.testing.assert_close(with_channel[0, :5], without[0, :5])   # causal: earlier positions unchanged
    assert not torch.allclose(with_channel[0, 5:], without[0, 5:])
