# CLAUDE.md

Bu dosya Claude Code'un her oturumda okuduğu proje belleğidir. Kısa tutulur; ayrıntılar `docs/` altındadır.

## Proje

Türkçe metindeki yazım ve dilbilgisi hatalarını bulan, düzelten ve ihlal edilen TDK kuralını gösteren bir model ile demo. Üniversite bitirme (capstone) projesi; tek kişi yürütüyor.

Hedef: Türkçe yazım hatalarını düzelten, hafif (yaklaşık 0,2–2,4B parametre) ve iyi genellenen bir model. Üç gerçek test seti hedef değil, genellenebilirliğin ölçüsüdür.

Araştırma soruları (K3, 2026-11-23 notu): (1) Ücretsiz bir T4'te QLoRA ile ince ayarlanan hafif modeller üç gerçek test setinde 2023 etiketleyicisine ve 2024 mT5 modeline ne kadar yaklaşıyor ya da onları geçiyor; model küçüldükçe ne kaybediliyor? (2) Sentetik veriye (GECTurk) organik yazım hataları (OSCAR GEC) ve keşif fazındaki teknikleri eklemek görülmemiş alanlara geçişi nasıl etkiliyor?

## Önce bunları oku

1. `docs/00_PROJE_OZETI.md`: ne yapıyoruz
2. `docs/01_KARARLAR.md`: alınan kararlar ve nedenleri. Bir kararı değiştirmeden önce kullanıcıya sor.
3. `docs/03_TEKNIK_NOTLAR.md`: bilinen tuzaklar
4. `docs/gorevler/`: sıradaki görev. `docs/GUNLUK.md` hangi görevde olduğumuzu söyler.

## Değişmez kurallar

- **Kapı A:** Yayımlanmış temel skorlar yeniden üretilmeden model eğitimi kodu Kaggle'da çalıştırılmaz.
- **Kapı B:** Ana model, veri boyutu, `max_len`, değerlendirme protokolü ve keşif fazının teknik listesi seçildikten sonra değişmez (K20 notu).
- **Keşif fazı sınırlıdır (K25):** en fazla üç teknik, yalnızca doğrulama setlerinde. Her yapılandırma test setlerinde bir kez ölçülür.
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

Güncel durum ve sıradaki adım için `docs/GUNLUK.md` dosyasının en üstündeki kayda bak. Görev 01'de yalnızca kullanıcının elle kontrolü kaldı; Kapı A geçildi (2026-11-23). Görev 03'ün yerel kısmı bitti (eğitim ve çıkarım betikleri, duman testi); sıradaki iş Kaggle'da deneme koşusu ve sekiz adayın pilotu (`docs/gorevler/03_pilot_ve_model_secimi.md`, adım 6).
