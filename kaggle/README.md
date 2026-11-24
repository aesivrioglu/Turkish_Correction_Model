# Kaggle

`calistirici.ipynb` ince bir çalıştırıcıdır: depoyu klonlar, bağımlılıkları kurar ve bir betik çalıştırır. Buraya mantık yazılmaz.

Kurulum ve kullanım: [docs/04_ORTAM_VE_IS_AKISI.md](../docs/04_ORTAM_VE_IS_AKISI.md)

Hatırlatmalar:

- Eğitimi "Save Version" ile toplu modda çalıştır; etkileşimli oturum boştayken de kota harcar.
- Çalıştırmadan önce `RUNS` listesini ve `GITHUB_USER` değerini kontrol et. `RUNS` içindeki her satır `scripts/run_pilot.py`'nin bir çağrısıdır (yapılandırma, adaylar, `--modes`, `--zero-shot`); ikinci ve sonraki satırlarda `--append` olmalı.
- Veri depoda yok: `trgec-pilot-veri` adlı özel Kaggle veri seti not defterine eklenmiş olmalı (kurulum belgesinde 7. adım).
- Çıktı: Output bölümündeki `pilot_outputs.zip`. Skorlama yerelde yapılır.
- Kapı A geçilmeden eğitim çalıştırılmaz (2026-11-23'te geçildi).
