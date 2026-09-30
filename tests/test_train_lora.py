"""The memory-saving reply loss must equal the standard masked causal-LM loss (tiny random model, CPU)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from src.models.train_lora import TARGET_MODULES, reply_loss  # noqa: E402


def tiny_qwen3():
    torch.manual_seed(0)
    config = transformers.Qwen3Config(vocab_size=500, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                      max_position_embeddings=512)
    return transformers.Qwen3ForCausalLM(config)


@pytest.mark.parametrize("with_lora", [False, True])
def test_reply_loss_equals_standard_masked_loss(with_lora: bool) -> None:
    model = tiny_qwen3()
    if with_lora:
        model = peft.get_peft_model(model, peft.LoraConfig(r=4, lora_alpha=8, target_modules=TARGET_MODULES,
                                                           task_type="CAUSAL_LM"))
    ids = torch.randint(0, 500, (60,)).tolist()
    labels = [-100] * 45 + ids[45:]  # a long prompt without loss, then a short reply
    reference = model(input_ids=torch.tensor([ids]), labels=torch.tensor([labels])).loss
    loss = reply_loss(model, ids, labels)
    assert loss.item() == pytest.approx(reference.item(), rel=1e-5)
    if with_lora:
        loss.backward()
        lora_grads = [p.grad for n, p in model.named_parameters() if "lora_" in n]
        assert lora_grads and all(g is not None for g in lora_grads)  # only LoRA weights learn
        assert all(p.grad is None for n, p in model.named_parameters() if "lora_" not in n)
