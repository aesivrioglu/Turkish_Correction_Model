# Günlük

Her çalışma oturumunun sonunda en üste yeni bir kayıt eklenir. Amaç, bir sonraki oturumun (ve Claude Code'un) kaldığı yerden devam edebilmesidir.

Kayıt biçimi:

```
## YYYY-AA-GG, görev NN
- Yapılan:
- Ölçülen (sayılarla):
- Karar ya da değişiklik (varsa K numarası):
- Sorun ya da açık soru:
- Sıradaki adım:
```

## 2026-11-13, görev 01 (adım 2: tokenizasyon)

- Yapılan:
  - OSCAR testi (`04cd0c3`) ve tweet seti (`06e9476`) sabit sürümle `data/raw/` altına indirildi.
  - İki setin ve ERRANT-TR'nin tokenizasyonu incelendi (`raporlar/01_tokenizasyon.md`).
  - `normalize_for_match` içindeki birleşen nokta hatası düzeltildi ve testi eklendi.
- Ölçülen:
  - OSCAR testi: 2.408 satır; `inp` ile M2 `S` satırı yalnızca 291 satırda aynı; hatasız satır 26 (%1,1); ortalama 25 token.
  - Tweetler: 2.000 satır; hatasız 89 (%4,5); `|` işaretli 425 satır; 47 birleşen nokta.
  - Testler: 31 geçti.
- Karar ya da değişiklik: Yok. "Ortak biçimde doğal metin" önerisi kullanıcı onayı bekliyor.
- Sorun ya da açık soru:
  - GECTurk M2 dosyaları hâlâ yok; M2 `S` satırı ile parquet tokenlarının birebir aynı olduğu doğrulanamadı.
  - Ersoy ve Yıldız'ın tweet çıktılarından hangi noktalamayı sildiği belirsiz (Kapı A riski).
- Sıradaki adım: Onay gelince adım 3: OSCAR testi ve tweet dönüştürücüleri, sahte veriyle testleri.

## 2026-11-13, görev 01 (karar güncellemesi)

- Yapılan:
  - K15'e tarihli not eklendi.
  - `00_PROJE_OZETI.md` içindeki temel tablo ve başarı ölçütü düzeltildi.
  - Görev 02, iş planı ve `base.yaml` (`eval.scorers`) yeni karara uyduruldu.
  - Koddaki yanlış karar atıfları düzeltildi (K12 → K15, K13 → K14/K16).
- Ölçülen: Yok.
- Karar ya da değişiklik:
  - K15 güncellendi: OSCAR ve tweetlerde ERRANT-TR; film yorumlarında M2 scorer ve ERRANT-TR birlikte.
  - OSCAR testinde hedef 55,4 (OSCAR GEC mT5). 62,8 (GPT GEC mT5) ikinci referans.
- Sorun ya da açık soru: GECTurk M2 dosyaları henüz `data/raw/gecturk/` altında değil.
- Sıradaki adım: Görev 01, adım 2 (tokenizasyon) ve adım 3 (dönüştürücüler).

## 2026-11-12, görev 01 (adım 1)

- Yapılan:
  - Depo GitHub'a gönderildi (`aesivrioglu/Turkish_Correction_Model`, herkese açık) ve `.venv` kuruldu.
  - Üç belgedeki bozuk UTF-8 baytı düzeltildi.
  - Doğrulanmamış varsayımlar araştırıldı (`raporlar/00_varsayim_dogrulama.md`).
  - Veri kaynakları incelendi (`raporlar/01_veri_kaynaklari.md`).
  - `base.yaml` içindeki Kumru kimliği `vngrs/Kumru-2B` olarak düzeltildi ve OSCAR kimlikleri eklendi.
  - `05_KAYNAKLAR.md` içindeki yayımlanmış sayılar düzeltildi.
- Ölçülen:
  - GECTurk ayrımları: train 96.919, dev 20.769, test 20.769, movie_reviews 300. Hepsinde hatasız cümle oranı yaklaşık %50.
  - GECTurk'te tek başına noktalama tokenı oranı %0,16; noktalama kelimeye bitişik.
  - Train'de 19 numaralı kuraldan 3, 21 numaralı kuraldan 38 cümle var.
  - Tweet seti 2.000 satır.
- Karar ya da değişiklik: Yok. Önerilen değişiklikler kullanıcı onayı bekliyor (K15 skorlayıcı, OSCAR testinde 55,4 ile 62,8).
- Sorun ya da açık soru:
  - GECTurk'ün HF kopyasında hedef cümle yok; M2 dosyaları Google Drive'da ve kullanıcının indirmesi gerekiyor.
  - Tweet seti ve mT5 ağırlıklarının lisansı yok; danışmana sorulacak.
  - Kumru 2,38B parametreyle K6'daki aralığın biraz üstünde.
- Sıradaki adım: M2 dosyaları `data/raw/gecturk/` altına gelince görev 01, adım 2 (tokenizasyon) ve adım 3 (dönüştürücüler).

## 2026-11-12, başlangıç

- Yapılan: Depo iskeleti, yardımcı modüller (`src/trgec`) ve testler hazırlandı. Kararlar, plan ve teknik notlar `docs/` altına yazıldı.
- Ölçülen: 30 test geçiyor. GECTurk veri klasörü 12 MB.
- Karar ya da değişiklik: K1 ile K23 arası kararlar kaydedildi.
- Sorun ya da açık soru: Danışman onayı ve teslim tarihleri bekleniyor.
- Sıradaki adım: `gorevler/01_veri_hazirligi.md`
