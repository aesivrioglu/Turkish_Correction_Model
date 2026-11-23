"""Ham veri kaynaklarını ortak JSONL biçimine çeviren dönüştürücüler.

Ortak biçim (data/processed/<ad>.jsonl, satır başına bir örnek):
    {"id": str, "source": str, "target": str, "rules": [str, ...],
     "origin": "gecturk" | "oscar" | "tweets", "split": str, "extra": {...}}

K24: source ve target doğal metindir. Tokenizasyon, küçük harfe çevirme ya da
noktalama silme uygulanmaz; ölçüte özgü dönüşümler değerlendirme katmanında
yapılır. Özgün etiketler "extra" alanında saklanır.

İşlevler dosya yolu değil metin alır; böylece küçük sahte veriyle sınanabilir.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from trgec.text import squeeze_spaces

ORIGIN_GECTURK = "gecturk"
ORIGIN_OSCAR = "oscar"
ORIGIN_TWEETS = "tweets"

TWEET_COLUMNS = ("raw", "preprocessed", "corrected", "annotated")
TWEET_HEADER_START = "RAW DATA"
TWEET_SPACE_MARK = "|"  # veri setinde kelime hizasını korumak için kullanılan boşluk işareti
M2_NOOP = "noop"

_BLOCK_SEP_RE = re.compile(r"\n[ \t]*\n")


def make_record(
    index: int,
    origin: str,
    split: str,
    source: str,
    target: str,
    rules: Sequence[str] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ortak biçimde tek bir örnek üretir. Kimlik 1'den başlar: 'oscar-test-000001'."""
    return {
        "id": f"{origin}-{split}-{index:06d}",
        "source": source,
        "target": target,
        "rules": list(rules),
        "origin": origin,
        "split": split,
        "extra": dict(extra or {}),
    }


def split_lines(text: str) -> list[str]:
    """Metni yalnızca satır sonu karakterinden böler (LF ve CRLF).

    str.splitlines() kullanılmaz: U+2028 ve U+0085 gibi karakterlerden de böler.
    Web metninde bunlar cümlenin içinde geçebilir ve satır hizasını kaydırır.
    """
    lines = [line[:-1] if line.endswith("\r") else line for line in text.split("\n")]
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def m2_edit_types(m2_text: str) -> list[list[str]]:
    """M2 metnindeki her blok için düzenleme türlerini (ör. 'R:SPELL') döndürür.

    Blok biçimi: bir 'S ...' satırı, ardından sıfır ya da daha çok
    'A başlangıç bitiş|||tür|||düzeltme|||...' satırı. Bloklar boş satırla ayrılır.
    """
    result: list[list[str]] = []
    normalized = m2_text.replace("\r\n", "\n").strip("\n")
    if not normalized.strip():
        return result
    for number, block in enumerate(_BLOCK_SEP_RE.split(normalized), start=1):
        lines = block.split("\n")
        if not lines[0].startswith("S "):
            raise ValueError(f"M2 blok {number}: 'S ' satırıyla başlamıyor")
        types: list[str] = []
        for line in lines[1:]:
            if not line.startswith("A "):
                raise ValueError(f"M2 blok {number}: beklenmeyen satır: {line[:40]!r}")
            fields = line.split("|||")
            if len(fields) < 2:
                raise ValueError(f"M2 blok {number}: bozuk A satırı: {line[:40]!r}")
            if fields[1] != M2_NOOP:
                types.append(fields[1])
        result.append(types)
    return result


def parse_m2(m2_text: str) -> list[tuple[str, list[dict[str, Any]]]]:
    """M2 metnini bloklara ayırır: her blok için (kaynak cümle, düzenlemeler).

    Düzenleme: {"start", "end", "rule", "correction"}. start ve end, kaynak
    cümlenin boşlukla bölünmüş kelimelerindeki konumlardır (end dahil değil).
    'A -1 -1|||noop' satırları ("hata yok" işareti) atlanır.
    """
    blocks: list[tuple[str, list[dict[str, Any]]]] = []
    normalized = m2_text.replace("\r\n", "\n").strip("\n")
    if not normalized.strip():
        return blocks
    for number, block in enumerate(_BLOCK_SEP_RE.split(normalized), start=1):
        lines = block.split("\n")
        if not lines[0].startswith("S "):
            raise ValueError(f"M2 blok {number}: 'S ' satırıyla başlamıyor")
        edits: list[dict[str, Any]] = []
        for line in lines[1:]:
            fields = line.split("|||")
            position = fields[0].split(" ")
            if not line.startswith("A ") or len(fields) < 3 or len(position) != 3:
                raise ValueError(f"M2 blok {number}: bozuk A satırı: {line[:40]!r}")
            if fields[1] == M2_NOOP:
                continue
            edits.append({"start": int(position[1]), "end": int(position[2]), "rule": fields[1], "correction": fields[2]})
        blocks.append((lines[0][2:], edits))
    return blocks


def apply_m2_edits(source: str, edits: Sequence[dict[str, Any]]) -> str:
    """Düzenlemeleri kaynak cümleye uygular ve düzeltilmiş cümleyi döndürür.

    Cümle boşluklardan kelimelere bölünür; her düzenleme start..end arasındaki
    kelimelerin yerine düzeltmeyi koyar. Düzenlemeler sondan başa doğru
    uygulanır: baştaki bir değişiklik kelime sayısını değiştirirse sonrakilerin
    konumu kaymasın diye. Örtüşen ya da cümlenin dışına taşan düzenleme hatadır.
    """
    tokens = source.split()
    previous_start = len(tokens) + 1
    for edit in sorted(edits, key=lambda e: (e["start"], e["end"]), reverse=True):
        start, end = edit["start"], edit["end"]
        if not 0 <= start <= end <= len(tokens) or end > previous_start:
            raise ValueError(f"Geçersiz ya da örtüşen düzenleme: {start}-{end} ({len(tokens)} kelime)")
        tokens[start:end] = edit["correction"].split()
        previous_start = start
    return " ".join(tokens)


def resolve_overlapping_edits(edits: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Örtüşen düzenlemelerden birini seçer; örtüşmeyenlere dokunmaz.

    Film yorumlarında üç cümlede aynı kelimeye birden çok düzenleme yazılmış:
    aynı düzenlemenin yinelenmesi, aynı kelimenin iki adımda düzeltilmesi
    ('izlenilmeseda' -> 'izlenilmesede' -> 'izlenilmese de') ve iç içe iki
    düzenleme ('halbu' -> 'Halbu' ile 'halbu ki' -> 'Halbuki'). Tek bir hedef
    cümle üretmek için kural: örtüşenlerden en geniş olan, genişlik eşitse
    dosyada sonra yazılan tutulur. Üç cümlede de bu, son ve tam düzeltmedir.
    """
    kept: list[dict[str, Any]] = []
    by_priority = sorted(enumerate(edits), key=lambda pair: (pair[1]["end"] - pair[1]["start"], pair[0]), reverse=True)
    for _, edit in by_priority:
        if all(other["end"] <= edit["start"] or other["start"] >= edit["end"] for other in kept):
            kept.append(edit)
    return sorted(kept, key=lambda edit: edit["start"])


def convert_gecturk(m2_text: str, split: str, target_text: str | None = None) -> list[dict[str, Any]]:
    """GECTurk M2 dosyasını ortak biçime çevirir.

    source: 'S' satırındaki cümle, olduğu gibi. target: hedef dosyası verildiyse
    onun aynı sıradaki satırı; verilmediyse (film yorumları) düzenlemelerin
    kaynağa uygulanmış hali (örtüşenler resolve_overlapping_edits ile çözülür;
    böyle cümlelerde extra['overlap_resolved'] True olur). rules: cümlede ihlal
    edilen kurallar ('rule_6'), ilk geçiş sırasıyla ve yinelemesiz; hatasız
    cümlede boş liste. extra['edits']: dosyadaki bütün düzenlemelerin konumu,
    yanlış ve doğru ifadesi ve kuralı; eğitimde düzeltme listesi buradan kurulur.
    """
    blocks = parse_m2(m2_text)
    targets = split_lines(target_text) if target_text is not None else None
    if targets is not None and len(targets) != len(blocks):
        raise ValueError(f"GECTurk {split}: M2'de {len(blocks)} cümle var ama hedef dosyasında {len(targets)} satır var")
    records = []
    for i, (source, edits) in enumerate(blocks):
        tokens = source.split()
        extra: dict[str, Any] = {}
        if targets is not None:
            target = targets[i]
        else:
            usable = resolve_overlapping_edits(edits)
            target = apply_m2_edits(source, usable)
            if len(usable) != len(edits):
                extra["overlap_resolved"] = True
        detailed = [
            {
                "start": edit["start"],
                "end": edit["end"],
                "source": " ".join(tokens[edit["start"] : edit["end"]]),
                "target": edit["correction"],
                "rule": edit["rule"],
            }
            for edit in edits
        ]
        rules = list(dict.fromkeys(edit["rule"] for edit in edits))
        extra["edits"] = detailed
        records.append(make_record(i + 1, ORIGIN_GECTURK, split, source, target, rules=rules, extra=extra))
    return records


def convert_oscar_test(inp_text: str, ref_text: str, m2_text: str | None = None) -> list[dict[str, Any]]:
    """OSCAR GEC test setini ortak biçime çevirir.

    source: inp.txt satırı, target: ref.txt satırı; ikisi de olduğu gibi kalır.
    M2 metni verilirse her satırın düzenleme türleri extra['m2_types'] içine yazılır.
    Resmi skorlama data/raw altındaki özgün M2 dosyasını kullanır, bu alanı değil.
    """
    sources = split_lines(inp_text)
    targets = split_lines(ref_text)
    if len(sources) != len(targets):
        raise ValueError(f"OSCAR test: inp {len(sources)} satır, ref {len(targets)} satır")
    types = m2_edit_types(m2_text) if m2_text is not None else None
    if types is not None and len(types) != len(sources):
        raise ValueError(f"OSCAR test: {len(sources)} satır var ama M2'de {len(types)} blok var")
    records = []
    for i, (source, target) in enumerate(zip(sources, targets)):
        extra: dict[str, Any] = {}
        if types is not None:
            extra["m2_types"] = types[i]
        records.append(make_record(i + 1, ORIGIN_OSCAR, "test", source, target, extra=extra))
    return records


def oscar_row_problem(sentence: Any, correction: Any, max_words: int) -> str | None:
    """OSCAR eğitim satırı alt kümeye girebilir mi? Giremezse nedenini döndürür.

    Nedenler: 'empty' (alan boş ya da metin değil), 'multiline' (alanın içinde
    satır sonu var; test setleri tek satırlıktır), 'too_long' (kaynak cümle
    max_words kelimeden uzun; K11'in 2026-11-23 notu). Sorun yoksa None döner.
    """
    if not isinstance(sentence, str) or not isinstance(correction, str):
        return "empty"
    sentence, correction = sentence.strip(), correction.strip()
    if not sentence or not correction:
        return "empty"
    if any(mark in text for text in (sentence, correction) for mark in ("\n", "\r")):
        return "multiline"
    if len(sentence.split()) > max_words:
        return "too_long"
    return None


def oscar_train_record(row_number: int, sentence: str, correction: str, split: str) -> dict[str, Any]:
    """OSCAR eğitim verisinin bir satırını ortak biçime çevirir.

    Kimlik, satırın train.csv içindeki sıra numarasını taşır; böylece alt küme
    kimlik listesinden yeniden üretilebilir. Metnin yalnızca baş ve sonundaki
    boşluk silinir; içi olduğu gibi kalır (K24).
    """
    return make_record(row_number, ORIGIN_OSCAR, split, sentence.strip(), correction.strip(), extra={"row": row_number})


def oscar_clean_record(row_number: int, correction: str, split: str) -> dict[str, Any]:
    """OSCAR satırının düzeltilmiş halinden hatasız bir örnek üretir.

    OSCAR eğitim verisinde kaynak ve hedefi aynı olan satır neredeyse yoktur
    (%0,08). Modelin doğru yazılmış organik metne dokunmamayı öğrenmesi için
    düzeltilmiş cümle hem kaynak hem hedef olarak kullanılır (K27). Kimliğin
    sonundaki '-c' ve extra['clean_from_target'] bu örnekleri ayırt eder.
    """
    text = correction.strip()
    record = make_record(row_number, ORIGIN_OSCAR, split, text, text, extra={"row": row_number, "clean_from_target": True})
    record["id"] += "-c"
    return record


def convert_tweets(text: str) -> list[dict[str, Any]]:
    """#Turki$hTweets dosyasını (data.txt, sekmeyle ayrılmış) ortak biçime çevirir.

    source: PREPROCESSED sütunu, target: CORRECTED sütunu. İki sütunda da '|'
    işareti boşluğa çevrilir ('tabii|ki' -> 'tabii ki'). Dosyadan gelen baş ve
    sondaki boşluklar ile çift boşluklar temizlenir; bu sütunlar boşlukla
    ayrılmış kelime dizisidir, boşluk içerik taşımaz. Harfler ve işaretler
    değişmez (kaynaktaki birleşen noktalar dahil). Ham tweet, kelime başına
    etiketler ve '|' işaretli özgün biçimler extra içinde saklanır.
    """
    lines = split_lines(text.lstrip("\ufeff"))
    if not lines or not lines[0].startswith(TWEET_HEADER_START):
        raise ValueError("Tweet dosyası: başlık satırı bulunamadı")
    records = []
    for number, line in enumerate(lines[1:], start=2):
        columns = line.split("\t")
        if len(columns) != len(TWEET_COLUMNS):
            raise ValueError(f"Tweet satırı {number}: {len(TWEET_COLUMNS)} sütun bekleniyordu, {len(columns)} bulundu")
        raw, preprocessed, corrected, annotated = columns
        labels = annotated.split()
        if len(labels) != len(preprocessed.split()):
            raise ValueError(
                f"Tweet satırı {number}: {len(preprocessed.split())} kelime var ama {len(labels)} etiket var"
            )
        records.append(
            make_record(
                number - 1,
                ORIGIN_TWEETS,
                "test",
                squeeze_spaces(preprocessed.replace(TWEET_SPACE_MARK, " ")),
                squeeze_spaces(corrected.replace(TWEET_SPACE_MARK, " ")),
                extra={
                    "raw": raw,
                    "labels": labels,
                    "source_aligned": preprocessed,
                    "target_aligned": corrected,
                },
            )
        )
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    """Örnekleri satır başına bir JSON olarak yazar; yazılan satır sayısını döndürür."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """write_jsonl ile yazılmış dosyayı okur."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle.read().split("\n") if line]
