from types import SimpleNamespace

import pytest

from trgec.modeling import restore_tied_output_layer

torch = pytest.importorskip("torch")


class TinyModel:
    """İki tablosu olan sahte dil modeli: gömme (6 token x 4 boyut) ve çıktı katmanı."""

    def __init__(self, tie: bool, zero_head: bool) -> None:
        self.config = SimpleNamespace(tie_word_embeddings=tie)
        self.embed = torch.nn.Embedding(6, 4)
        self.head = torch.nn.Linear(4, 6, bias=False)
        if zero_head:
            torch.nn.init.zeros_(self.head.weight)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head


def test_empty_output_layer_is_tied_to_embeddings():
    model = TinyModel(tie=True, zero_head=True)
    assert restore_tied_output_layer(model) is True
    assert model.head.weight is model.embed.weight
    # Artık puanlar sabit değil: model tokenları ayırt edebiliyor.
    logits = model.head(torch.ones(4))
    assert float(logits.max() - logits.min()) > 0
    # İkinci çağrı bir şey yapmaz (zaten bağlı).
    assert restore_tied_output_layer(model) is False


def test_healthy_or_untied_models_are_left_alone():
    healthy = TinyModel(tie=True, zero_head=False)
    before = healthy.head.weight.clone()
    assert restore_tied_output_layer(healthy) is False
    assert torch.equal(healthy.head.weight, before)
    # Yapılandırma bağlama istemiyorsa sıfır tabloya da dokunulmaz.
    assert restore_tied_output_layer(TinyModel(tie=False, zero_head=True)) is False
