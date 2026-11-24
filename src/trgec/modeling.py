"""Model yüklendikten sonra yapılan küçük onarımlar (eğitim ve çıkarım ortak kullanır).

Bu modül torch'u içe aktarmaz; yalnızca kendisine verilen model nesnesiyle çalışır.
"""

from __future__ import annotations

from typing import Any


def restore_tied_output_layer(model: Any) -> bool:
    """Çıktı katmanı boş (tamamı sıfır) yüklenmişse onu gömme katmanına bağlar.

    Bir dil modelinde iki büyük tablo vardır: gömme katmanı (token -> vektör) ve
    çıktı katmanı (vektör -> her token için puan). Birçok modelde ikisi aynı
    tablodur ("ağırlık bağlama"); yapılandırmada tie_word_embeddings=True yazar.

    Mecellem-Qwen3-1.7B-TR'nin ağırlık dosyasında çıktı katmanı ayrıca kayıtlı ve
    tamamı sıfır. transformers 5.x iki tabloyu farklı görünce bağlamıyor; model
    her tokena eşit puan veriyor: kayıp ln(sözlük boyutu) = 11,93'te kalıyor,
    gradyan sıfır oluyor, çıktı hep aynı token oluyor (pilotta ölçüldü).

    Onarım yalnızca üç koşul birlikte sağlanırsa yapılır: yapılandırma bağlama
    istiyor, iki tablo ayrı ve çıktı tablosunun tamamı sıfır. Sağlam yüklenen
    modellere dokunulmaz. Döndürür: onarım yapıldı mı?
    """
    if not getattr(model.config, "tie_word_embeddings", False):
        return False
    inputs, outputs = model.get_input_embeddings(), model.get_output_embeddings()
    if inputs is None or outputs is None or outputs.weight is inputs.weight:
        return False
    # Nicemlenmiş (4-bit) bir katmanın ağırlığı paketlenmiş tam sayılardır; biçimi tutmaz, dokunulmaz.
    if outputs.weight.shape != inputs.weight.shape or not outputs.weight.is_floating_point():
        return False
    if bool((outputs.weight != 0).any()):
        return False
    outputs.weight = inputs.weight
    return True
