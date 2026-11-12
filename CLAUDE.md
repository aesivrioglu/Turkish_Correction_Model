# CLAUDE.md

Bu dosya Claude Code'un her oturumda okuduğu proje belleğidir. Kısa tutulur; ayrıntılar `docs/` altındadır.

## Proje

Türkçe metindeki yazım ve dilbilgisi hatalarını bulan, düzelten ve ihlal edilen TDK kuralını gösteren bir model ile demo. Üniversite bitirme (capstone) projesi; tek kişi yürütüyor.

Araştırma sorusu: Sentetik veriyle (GECTurk) eğitilmiş küçük bir LLM, organik yazım hatalarıyla (OSCAR GEC) desteklendiğinde üç gerçek test setinde 2023 etiketleyicisini ve 2024 mT5 modelini geçebiliyor mu?

## Önce bunları oku

1. `docs/00_PROJE_OZETI.md`: ne yapıyoruz
2. `docs/01_KARARLAR.md`: alınan kararlar ve nedenleri. Bir kararı değiştirmeden önce kullanıcıya sor.
3. `docs/03_TEKNIK_NOTLAR.md`: bilinen tuzaklar
4. `docs/gorevler/`: sıradaki görev. `docs/GUNLUK.md` hangi görevde olduğumuzu söyler.

## Değişmez kurallar

- **Kapı A:** Yayımlanmış temel skorlar yeniden üretilmeden model eğitimi kodu Kaggle'da çalıştırılmaz.
- **Kapı B:** Model ve veri boyutu seçildikten sonra değişmez.
- **Test setlerine dokunma:** Film yorumları, OSCAR test ve tweet setleri yalnızca nihai değerlendirmede kullanılır. Eşik, model ve hiperparametre seçimi doğrulama setlerinde yapılır.
- **Skorlayıcı tutarlılığı:** Her test setinde, o setin yayımlanmış skorunda kullanılan araç kullanılır. Kendi yazdığımız fark hesapları (`trgec.edits`) yalnızca iç kullanım içindir.
- **Türkçe harfler:** Karşılaştırmalarda `str.lower()` değil `trgec.text.tr_lower` kullan.
- **Varsayma, doğrula:** Veri sütun adlarını, Hugging Face kimliklerini ve lisansları tahmin etme. Önce incele, bulduğunu `docs/raporlar/` altına yaz.
- **Gizli bilgi ve veri depoya girmez:** `.env`, `kaggle.json`, `data/raw`, `data/processed`, `outputs`.
- **GPU yalnızca Kaggle'da:** Yerelde her şey CPU ile çalışmalı. Eğitim betiği `--smoke-test` ile yerelde denenmeden Kaggle'a gönderilmez.

## Çalışma biçimi

- Kullanıcı bu kodu jüri önünde savunacak. Yazdığın her önemli parçayı kısaca açıkla; kullanıcının anlamadan onayladığı kod bırakma.
- Küçük adımlarla ilerle, her adımda `pytest -q` geçsin, anlamlı commit mesajı yaz.
- Yeni bir bağımlılık eklemeden ya da bir kararı değiştirmeden önce sor.
- Her oturumun sonunda `docs/GUNLUK.md` dosyasına ne yapıldığını, ne ölçüldüğünü ve sıradaki adımı yaz.
- Belirsizlik varsa tahmin yürütme; seçenekleri kullanıcıya sun.

## Komutlar

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest -q                                            # tüm testler
python scripts/prepare_data.py --config configs/base.yaml
python scripts/train.py --config configs/d1_gecturk.yaml --smoke-test
```

## Yapı

```
src/trgec/     ortak kod (biçimler, örnekleme, emniyet kemeri, bootstrap, sızıntı taraması)
scripts/       prepare_data, train, predict, evaluate (komut satırı betikleri)
configs/       base.yaml + deney yapılandırmaları (D1, D2)
kaggle/        ince çalıştırıcı not defteri
docs/          kararlar, plan, teknik notlar, görevler, günlük
tests/         pytest testleri
```

## Durum

Başlangıç: iskelet ve yardımcı modüller hazır, testler geçiyor. Sıradaki görev: `docs/gorevler/01_veri_hazirligi.md`.
