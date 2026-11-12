# Görev 03: Model seçimi ve pilot

**Hafta:** 3. **Tahmini süre:** 10 saat. **GPU:** yaklaşık 2 saat.

## Amaç

Eğitim ve çıkarım betiklerini yazmak, iki aday modeli kısa eğitimlerle kıyaslamak, hızı ölçmek ve Kapı B'de model, veri boyutu ve `max_len` değerini dondurmak.

## Başlamadan önce

`docs/01_KARARLAR.md` içinde K6 ile K12 arası, K20 ve K21 maddelerini oku.

## Adımlar

### 1. Adayları doğrula

`configs/base.yaml` içindeki iki adayın Hugging Face kimliğini, lisansını, parametre sayısını ve sohbet şablonunu kontrol et. Kimlik yanlışsa ya da daha uygun güncel bir sürüm varsa kullanıcıya sor. Gemma ailesi seçilmez (K12).

### 2. Eğitim örnekleri

`trgec.formats.build_messages` ile:

- GECTurk örnekleri `MODE_LABEL` ("düzelt ve etiketle")
- OSCAR örnekleri `MODE_CORRECT` ("yalnızca düzelt")

Düzeltme listesi için kaynak ve hedef ifadeler GECTurk'ün M2 dosyalarındaki düzenlemelerden alınır.

### 3. Token uzunluğu

Her aday için eğitim örneklerinin (şablon uygulanmış tam metin) token uzunluklarını ölç. `trgec.lengths.suggest_max_len` ile %95'lik dilime göre `max_len` öner; `truncation_rate` ile kesilecek örnek oranını raporla. Ayrıca kelime başına ortalama token sayısını raporla: Türkçeyi verimli tokenize eden model daha hızlı eğitilir.

### 4. `scripts/train.py`

- Yapılandırmadan okur; 4-bit NF4, LoRA (r, alpha, dropout yapılandırmada), fp16
- Kayıp yalnızca cevap tokenlarında
- `--smoke-test`: 8 örnek, 2 adım, nicemleme kapalı, CPU
- Düzenli aralıklarla kayıp ve hız günlüğü; kayıp NaN olursa durur
- Bitince adaptörü Hugging Face Hub'daki özel depoya yükler

### 5. `scripts/predict.py`

- Açgözlü kod çözme, sınırlı `max_new_tokens`
- `trgec.formats.parse_output` ile ayrıştırma; `well_formed` kaydedilir
- `trgec.guard.apply_guard` isteğe bağlı; kemerli ve kemersiz çıktı birlikte kaydedilir
- Toplu çıkarım (batch) ve sol dolgu

### 6. Mini eğitimler (Kaggle)

Kapı A geçildikten sonra: her adayı 2 bin örnekle (GECTurk ve OSCAR karışık) eğit. Ölç:

- Adım başına süre ve bellek kullanımı
- GECTurk dev ve OSCAR dev üzerinde F0.5
- Bozuk çıktı oranı
- Kayıp eğrisi (NaN var mı?)

### 7. Süre tahmini ve karar

Ölçülen hızdan D1 (12 bin) ve D2 (24 bin) için süre tahmini çıkar. D2 tek oturuma (12 saat) rahat sığmıyorsa veri boyutunu düşür ve kullanıcıya öner.

`docs/raporlar/03_pilot.md` dosyasına iki adayın karşılaştırma tablosunu ve önerini yaz. Seçimi kullanıcı yapar.

## Kapı B

Kullanıcı modeli seçti. `configs/base.yaml` içinde `model.chosen`, `training.max_len` ve örnekleme boyutları dolduruldu ve bundan sonra değişmez.

Aynı anda B planı ölçütleri (K21) kontrol edilir. Biri geçerliyse eğitime geçmeden kullanıcıya bildir.

## Bitti sayılma ölçütleri

- [ ] Duman testi yerelde geçiyor
- [ ] İki mini eğitim tamamlandı, pilot raporu yazıldı
- [ ] Kapı B kararları yapılandırmaya ve `docs/GUNLUK.md` dosyasına işlendi
- [ ] `requirements-kaggle.txt` çalışan sürümlerle sabitlendi
- [ ] `pytest -q` geçiyor
