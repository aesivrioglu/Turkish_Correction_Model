"""Modelin girdi ve çıktı biçimi.

İki talimat vardır (bkz. docs/01_KARARLAR.md, K9):
- MODE_LABEL:   düzelt ve her düzeltmeyi kural etiketiyle listele (GECTurk örnekleri)
- MODE_CORRECT: yalnızca düzelt (kural etiketi olmayan OSCAR örnekleri)

Çıktı JSON değil, kısa satır biçimidir (K8):

    <düzeltilmiş cümle>
    ###
    yanlış => doğru | KURAL
    yanlış => doğru | KURAL

Hata yoksa liste yerine tek satır "YOK" yazılır.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .text import squeeze_spaces

SEPARATOR = "###"
NO_EDIT = "YOK"
OTHER_RULE = "OTHER_SPELL"

MODE_LABEL = "label"
MODE_CORRECT = "correct"

INSTRUCTIONS = {
    MODE_LABEL: (
        "Aşağıdaki cümledeki yazım ve dilbilgisi hatalarını en az değişiklikle düzelt. "
        "Önce düzeltilmiş cümleyi yaz. Sonra '###' satırının altına her düzeltmeyi "
        "'yanlış => doğru | KURAL' biçiminde ayrı satırda listele. "
        "Hata yoksa cümleyi aynen yaz ve listeye yalnızca 'YOK' yaz."
    ),
    MODE_CORRECT: (
        "Aşağıdaki cümledeki yazım ve dilbilgisi hatalarını en az değişiklikle düzelt. "
        "Yalnızca düzeltilmiş cümleyi yaz. Hata yoksa cümleyi aynen yaz."
    ),
}


@dataclass(frozen=True)
class Edit:
    """Tek bir düzeltme: kaynak ifade, hedef ifade ve ihlal edilen kural."""

    source: str
    target: str
    rule: str = OTHER_RULE


@dataclass
class ParsedOutput:
    """Model çıktısının ayrıştırılmış hali."""

    corrected: str
    edits: list[Edit] = field(default_factory=list)
    well_formed: bool = True


def _check_mode(mode: str) -> None:
    if mode not in INSTRUCTIONS:
        raise ValueError(f"Bilinmeyen mod: {mode!r}. Geçerli modlar: {sorted(INSTRUCTIONS)}")


def build_prompt(sentence: str, mode: str) -> str:
    """Kullanıcı mesajının metnini üretir."""
    _check_mode(mode)
    return f"{INSTRUCTIONS[mode]}\n\nCümle: {squeeze_spaces(sentence)}"


def build_target(corrected: str, edits: list[Edit], mode: str) -> str:
    """Modelin üretmesi beklenen hedef metni üretir."""
    _check_mode(mode)
    corrected = squeeze_spaces(corrected)
    if mode == MODE_CORRECT:
        return corrected
    if not edits:
        return f"{corrected}\n{SEPARATOR}\n{NO_EDIT}"
    lines = [f"{e.source} => {e.target} | {e.rule}" for e in edits]
    return f"{corrected}\n{SEPARATOR}\n" + "\n".join(lines)


def build_messages(
    sentence: str,
    mode: str,
    corrected: str | None = None,
    edits: list[Edit] | None = None,
) -> list[dict[str, str]]:
    """Sohbet şablonuna verilecek mesaj listesi.

    corrected verilirse eğitim örneği (asistan cevabıyla birlikte),
    verilmezse çıkarım girdisi üretilir.
    """
    messages = [{"role": "user", "content": build_prompt(sentence, mode)}]
    if corrected is not None:
        messages.append(
            {"role": "assistant", "content": build_target(corrected, edits or [], mode)}
        )
    return messages


def parse_output(text: str, expect_edits: bool = True) -> ParsedOutput:
    """Model çıktısını ayrıştırır.

    Biçim bozuksa hata fırlatmaz; well_formed=False döner. Böylece bozuk çıktı
    oranı ayrı bir metrik olarak raporlanabilir.
    """
    lines = [ln.strip() for ln in text.strip().splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ParsedOutput(corrected="", well_formed=False)

    if SEPARATOR not in lines:
        # Ayraç yok: ilk satır düzeltilmiş cümle kabul edilir.
        return ParsedOutput(corrected=squeeze_spaces(lines[0]), well_formed=not expect_edits)

    sep_at = lines.index(SEPARATOR)
    corrected = squeeze_spaces(" ".join(lines[:sep_at]))
    well_formed = bool(corrected)
    edits: list[Edit] = []
    for line in lines[sep_at + 1 :]:
        if line == NO_EDIT:
            continue
        left, bar, rule = line.rpartition("|")
        if not bar:
            left, rule, well_formed = line, OTHER_RULE, False
        source, arrow, target = left.partition("=>")
        if not arrow:
            well_formed = False
            continue
        edits.append(Edit(source.strip(), target.strip(), rule.strip() or OTHER_RULE))
    return ParsedOutput(corrected=corrected, edits=edits, well_formed=well_formed)
