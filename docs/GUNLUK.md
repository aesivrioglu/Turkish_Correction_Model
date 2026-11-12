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
