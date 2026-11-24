import pytest

from trgec import examples
from trgec.formats import MODE_CORRECT, MODE_LABEL, parse_output

GECTURK_RECORD = {
    "id": "gecturk-train-000001",
    "source": "Dün saha da maç vardı.",
    "target": "Dün sahada maç vardı.",
    "rules": ["rule_6"],
    "origin": "gecturk",
    "split": "train",
    "extra": {"edits": [{"start": 1, "end": 3, "source": "saha da", "target": "sahada", "rule": "rule_6"}]},
}
GECTURK_CLEAN = {**GECTURK_RECORD, "source": "Hava güzel.", "target": "Hava güzel.", "rules": [], "extra": {"edits": []}}
OSCAR_RECORD = {
    "id": "oscar-train-000007",
    "source": "bugun hava cok guzel",
    "target": "Bugün hava çok güzel",
    "rules": [],
    "origin": "oscar",
    "split": "train",
    "extra": {"row": 7},
}


class TemplateTokenizer:
    """Sohbet şablonu olan bir tokenizer'ın yerine geçer (ChatML benzeri)."""

    chat_template = "var"
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}<end>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")


class WrappingTokenizer(TemplateTokenizer):
    """Cevabı istemden farklı saran şablon: tam metin istemle başlamaz."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        if add_generation_prompt:
            return f"[SORU]{messages[0]['content']}[CEVAP-BEKLENIYOR]"
        return f"[SORU]{messages[0]['content']}[CEVAP]{messages[1]['content']}"


class NoEndMarkerTokenizer(TemplateTokenizer):
    """Cevaptan sonra yalnızca satır sonu yazan şablon (Cosmos GPT-2 gibi)."""

    eos_token = "<|stop|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        text = f"### Kullanıcı:\n{messages[0]['content']}\n### Asistan:\n"
        return text if len(messages) == 1 else text + messages[1]["content"] + "\n"


class PlainTokenizer:
    chat_template = None
    eos_token = "</s>"


def test_gecturk_record_gets_label_mode_and_edit_list():
    messages = examples.training_messages(GECTURK_RECORD)
    assert examples.record_mode(GECTURK_RECORD) == MODE_LABEL
    assert "Cümle: Dün saha da maç vardı." in messages[0]["content"]
    assert messages[1]["content"] == "Dün sahada maç vardı.\n###\nsaha da => sahada | rule_6"
    # Hedef, çıkarımda kullanılan ayrıştırıcıyla geri okunabilmeli.
    parsed = parse_output(messages[1]["content"])
    assert parsed.well_formed and parsed.corrected == "Dün sahada maç vardı."
    assert [(e.source, e.target, e.rule) for e in parsed.edits] == [("saha da", "sahada", "rule_6")]


def test_clean_gecturk_record_says_no_edit():
    assert examples.training_messages(GECTURK_CLEAN)[1]["content"] == "Hava güzel.\n###\nYOK"


def test_oscar_record_gets_correct_only_mode():
    messages = examples.training_messages(OSCAR_RECORD)
    assert examples.record_mode(OSCAR_RECORD) == MODE_CORRECT
    assert messages[1]["content"] == "Bugün hava çok güzel"
    assert messages[0]["content"] != examples.training_messages(GECTURK_RECORD)[0]["content"].replace("Dün saha da maç vardı.", "bugun hava cok guzel")


def test_inference_messages_have_no_answer():
    messages = examples.inference_messages("bugun hava guzel")
    assert len(messages) == 1 and messages[0]["role"] == "user"


def test_render_with_template_splits_prompt_and_answer():
    messages = examples.training_messages(OSCAR_RECORD)
    rendered = examples.render(TemplateTokenizer(), messages)
    assert rendered.prompt.endswith("<end><assistant>")
    assert rendered.answer == "<assistant>Bugün hava çok güzel<end>"[len("<assistant>") :]
    assert rendered.add_special_tokens is False
    # Çıkarım girdisi eğitimdeki istemle birebir aynıdır.
    inference = examples.render(TemplateTokenizer(), examples.inference_messages("bugun hava cok guzel", MODE_CORRECT))
    assert inference.prompt == rendered.prompt and inference.answer is None


def test_render_falls_back_when_template_wraps_answer_differently():
    rendered = examples.render(WrappingTokenizer(), examples.training_messages(OSCAR_RECORD))
    assert rendered.prompt.endswith("[CEVAP-BEKLENIYOR]")
    assert rendered.answer == "Bugün hava çok güzel<eos>"


def test_render_without_template_uses_fallback():
    rendered = examples.render(PlainTokenizer(), examples.training_messages(OSCAR_RECORD))
    assert rendered.prompt.startswith("Kullanıcı: ") and rendered.prompt.endswith("\nAsistan:\n")
    assert rendered.answer == "Bugün hava çok güzel</s>"
    assert rendered.add_special_tokens is True
    assert "###" not in rendered.prompt


def test_mix_records_is_seeded_and_respects_limit():
    gecturk = [{"id": f"g{i}"} for i in range(10)]
    oscar = [{"id": f"o{i}"} for i in range(10)]
    d2 = examples.mix_records(gecturk, oscar, seed=42)
    assert len(d2) == 20 and d2 == examples.mix_records(gecturk, oscar, seed=42)
    assert [r["id"] for r in d2] != [r["id"] for r in gecturk + oscar]  # karıştırılmış
    assert len(examples.mix_records(gecturk, [], seed=42)) == 10  # D1
    mini = examples.mix_records(gecturk, oscar, limit_per_source=3, seed=42)
    assert sorted(r["id"] for r in mini) == ["g0", "g1", "g2", "o0", "o1", "o2"]


def test_drop_over_length_never_truncates():
    kept, dropped = examples.drop_over_length(["a", "b", "c"], [10, 600, 512], ceiling=512)
    assert kept == ["a", "c"] and dropped == ["b"]
    with pytest.raises(ValueError):
        examples.drop_over_length(["a"], [1, 2], ceiling=5)


def test_render_adds_eos_when_template_has_no_end_marker():
    rendered = examples.render(NoEndMarkerTokenizer(), examples.training_messages(OSCAR_RECORD))
    assert rendered.answer == "Bugün hava çok güzel<|stop|>"


def test_stop_text_is_what_the_template_writes_after_the_answer():
    assert examples.stop_text(TemplateTokenizer()) == "<end>"  # şablonun işareti, EOS değil
    assert examples.stop_text(NoEndMarkerTokenizer()) == "<|stop|>"
    assert examples.stop_text(PlainTokenizer()) == "</s>"
    assert examples.stop_text(WrappingTokenizer()) == "<eos>"


class CharTokenizer(PlainTokenizer):
    """Her karakteri bir token sayan sahte tokenizer; özel token istenirse başa 1 ekler."""

    def __call__(self, texts, add_special_tokens=True):
        prefix = [1] if add_special_tokens else []
        return {"input_ids": [prefix + [ord(ch) for ch in text] for text in texts]}


def test_encode_masks_prompt_and_keeps_answer_labels():
    rendered = [examples.Rendered("ab", "cd!", add_special_tokens=True), examples.Rendered("abcd", "e!", add_special_tokens=True)]
    first, second = examples.encode(CharTokenizer(), rendered)
    assert first["input_ids"] == [1, ord("a"), ord("b"), ord("c"), ord("d"), ord("!")]
    # İstem (özel token dahil) kayba girmez; cevabın bütün tokenları, bitiş işareti dahil, girer.
    assert first["labels"] == [-100, -100, -100, ord("c"), ord("d"), ord("!")]
    assert first["length"] == 6 and second["length"] == 7
    assert [label for label in second["labels"] if label != -100] == [ord("e"), ord("!")]
    assert examples.encode(CharTokenizer(), []) == []
    with pytest.raises(ValueError):
        examples.encode(CharTokenizer(), [examples.Rendered("ab", None, add_special_tokens=True)])


def test_pad_batch_pads_right_and_ignores_padding_in_loss():
    features = [
        {"input_ids": [5, 6, 7], "labels": [-100, 6, 7], "length": 3},
        {"input_ids": [5], "labels": [5], "length": 1},
    ]
    batch = examples.pad_batch(features, pad_token_id=0)
    assert batch["input_ids"] == [[5, 6, 7], [5, 0, 0]]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"] == [[-100, 6, 7], [5, -100, -100]]


def test_use_plain_prompt_switches_listed_models_to_fallback_and_eos():
    listed, other = TemplateTokenizer(), TemplateTokenizer()
    assert examples.use_plain_prompt(listed, "org/base-model", ["org/base-model"]) is True
    assert examples.use_plain_prompt(other, "org/chat-model", ["org/base-model"]) is False
    assert examples.use_plain_prompt(other, "org/chat-model", None) is False
    # Listedeki model: düz istem, cevap şablonun işaretiyle değil EOS ile biter.
    rendered = examples.render(listed, examples.training_messages(OSCAR_RECORD))
    assert rendered.prompt.startswith("Kullanıcı: ") and rendered.answer == "Bugün hava çok güzel<eos>"
    assert examples.stop_text(listed) == "<eos>"
    # Listede olmayan model şablonunu korur.
    assert examples.stop_text(other) == "<end>"
