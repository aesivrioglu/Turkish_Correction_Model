# İş planı ve iş yükü

Plan altı iş paketinden oluşur ve 8 hafta sürer. Yapay zeka desteğiyle yaklaşık 72 saat aktif çalışma ve 10 GPU saati (aralık 8 ile 14) gerektirir. Haftalık yük ortalama 9 saattir.

Takvim varsayımı: 1. hafta 16 Kasım 2026 Pazartesi başlar, 8. hafta 10 Ocak 2027'de biter. Teslim tarihleri netleşince kaydırılır.

## İş yükü

| İş paketi | Hafta | Aktif çalışma, YZ destekli (saat) | GPU (saat) | Çıktı |
| --- | --- | --- | --- | --- |
| 1. Kurulum ve veri | H1 | 10 | 0 | Ortak biçimde üç test seti, iki eğitim alt kümesi, veri raporu |
| 2. Değerlendirme hattı | H2 | 10 | 1 | Kapı A: mT5'in 79,0 skoru yeniden üretildi |
| 3. Model seçimi ve pilot | H3 | 10 | 2 | Kapı B: seçilen model, ölçülmüş hız ve `max_len` |
| 4. İki eğitim (D1, D2) | H4 ile H5 | 8 | 5 ile 11 | Üç sette F0.5 sonuç tablosu |
| 5. Analiz ve kural kartları | H6 | 12 | 1 | Kural bazında analiz, güven aralıkları, 25 kural kartı |
| 6. Demo, rapor, sunum | H7 ile H8 | 22 | 0 | Gradio demosu, rapor, sunum |
| **Toplam** | **8 hafta** | **~72** | **~10** | |

Rakamlar kaba tahmindir. D2, D1'in iki katı veri gördüğü için GPU süresinin çoğu oradadır. Kesin süre 3. haftadaki pilotta ölçülür. Yapay zeka en çok kod, betik, demo ve rapor taslağını hızlandırır; GPU süresini, hata ayıklamayı ve savunmada anlatılacak şeyleri anlama süresini kısaltmaz.

## Paketler

### 1. Kurulum ve veri (H1)

Görev dosyası: `gorevler/01_veri_hazirligi.md`

- [ ] Danışmanla sadeleştirilmiş kapsamı onayla
- [ ] Depoyu kur, Claude Code'u çalıştır, testleri geçir
- [ ] GECTurk, OSCAR GEC (eğitim ve test) ve #Turki$hTweets'i indir; şemalarını incele
- [ ] Lisansları ve Hugging Face kimliklerini doğrula
- [ ] Hepsini ortak JSONL biçimine çevir; özgün M2 dosyalarını skorlama için sakla
- [ ] Sızıntı taraması (bütün eğitim ve test çiftleri)
- [ ] OSCAR'dan 500 cümlelik doğrulama dilimi ayır
- [ ] Katmanlı örneklemeyle D1 ve D2 alt kümelerini üret
- [ ] Her setten 20 örneği elle kontrol et

### 2. Değerlendirme hattı (H2)

Görev dosyası: `gorevler/02_degerlendirme_hatti.md`

- [x] Ersoy ve Yıldız'ın hangi skorlayıcıyı kullandığını doğrula (ERRANT-TR; K15 notu)
- [x] ERRANT-TR'nin kullanılabilir olup olmadığını doğrula (kullanılabilir, MIT)
- [ ] ERRANT-TR'yi kur; OSCAR, tweetler ve film yorumlarında skorlayıcı olarak bağla
- [ ] mT5 modelini üç sette çalıştır; tweetlerde 79,0'ı yeniden üret
- [ ] M2 scorer'ı film yorumları için kur, zaman aşımı ekle
- [ ] GECTurk etiketleyicisini çalıştırmayı dene (en fazla yarım gün)
- [ ] Skorlayıcıdan cümle bazlı sayıları çıkar, bootstrap'a bağla

**Kapı A:** mT5'in tweetlerdeki skoru ±2 puan içinde tutuyor. Tutmuyorsa eğitime geçilmez, hat düzeltilir.

### 3. Model seçimi ve pilot (H3)

Görev dosyası: `gorevler/03_pilot_ve_model_secimi.md`

- [ ] Aday modellerin Hugging Face kimliklerini ve lisanslarını doğrula
- [ ] Eğitim örneklerini iki talimatlı biçime çevir
- [ ] Token uzunluklarını ölç, `max_len` değerini belirle
- [ ] `train.py` ve `predict.py` yaz; yerelde duman testini geçir
- [ ] İki adayı 2 bin örnekle mini eğitimden geçir, doğrulama setinde kıyasla
- [ ] Adım başına hızı ölç, D1 ve D2 için süre tahmini çıkar
- [ ] fp16'da NaN kayıp olup olmadığını izle

**Kapı B:** Model, veri boyutu ve `max_len` seçildi; bundan sonra değişmez. B planı ölçütleri (K21) burada kontrol edilir.

### 4. İki eğitim (H4 ile H5)

- [ ] D1 eğitimi; adaptör Hugging Face Hub'a
- [ ] D2 eğitimi; adaptör Hugging Face Hub'a
- [ ] Emniyet kemeri eşiğini doğrulama setinde ayarla
- [ ] İki modelin üç test setindeki tahminlerini üret
- [ ] Sonuç tablosunu doldur

GPU çalışırken 5. paketin betikleri hazırlanır; beklemeye gerek yoktur.

### 5. Analiz ve kural kartları (H6)

- [ ] Kural bazında recall (film yorumları)
- [ ] Hata türü bazında analiz (OSCAR ve tweetler)
- [ ] Emniyet kemeri açık ve kapalıyken aşırı düzeltme oranı
- [ ] Model çiftleri için bootstrap güven aralıkları
- [ ] Bozuk çıktı oranı
- [ ] 25 kural kartını TDK sayfalarından taslakla, elle doğrula
- [ ] 15 ile 20 başarısız örneği niteliksel olarak incele

### 6. Demo, rapor, sunum (H7 ile H8)

- [ ] Gradio demosu: hataların altı çizili, tıklayınca kural kartı açılıyor
- [ ] Hugging Face Spaces'te yayınla (ücretsiz katman CPU; gerekirse 4-bit GGUF)
- [ ] Rapor: giriş, literatür, veri, yöntem, sonuçlar, sınırlılıklar
- [ ] Figürler: üç sette model karşılaştırması, kural bazında recall
- [ ] Sunum ve canlı demo provası

## Yedek liste: zaman kalırsa

Çekirdekten çıkarılan işler. Çekirdek erken biterse yukarıdan aşağıya eklenir. Saatler kaba tahmindir.

| İş | Ne katar | Ek aktif saat | Ek GPU saati |
| --- | --- | --- | --- |
| Nihai modeli 3 tohumla tekrarlamak | Sonuçların tesadüf olmadığını gösterir | 3 | 10 |
| Büyük bir ticari LLM'i yalnızca 300 cümlede denemek | Üst sınırı gösterir | 4 | 0 (API ücreti) |
| D2 için eşit boyutlu kontrol (6 bin + 6 bin) | "Daha çok veri mi, organik veri mi?" sorusunu ayırır | 2 | 3 |
| BOUN de/da testinde ölçmek | Tek kuralda ek doğrulama | 2 | 0,5 |
| GPT GEC verisiyle üçüncü karışım (D3) | Veri sorusunu genişletir | 5 | 5 |
| İkinci bir küçük LLM'i eğitmek | Bulgunun modele bağlı olup olmadığını gösterir | 6 | 8 |
| 200 ile 300 cümlelik kendi test setimizi etiketlemek | Özgün veri katkısı, istatistiksel güç | 15 ile 20 | 0 |
| Serbest metin açıklama damıtma ve insan değerlendirmesi | Kural kartının ötesinde doğal açıklama | 20 | 12 |
