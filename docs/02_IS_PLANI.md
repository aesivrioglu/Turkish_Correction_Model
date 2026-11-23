# İş planı ve iş yükü

Plan altı iş paketinden oluşur ve 8 hafta sürer. 2026-11-23'teki kapsam genişletmesinden (K1 notu, K25, K26) sonra yapay zeka desteğiyle yaklaşık 90–100 saat aktif çalışma ve 20–25 GPU saati gerektirir (önceki tahmin 72 saat ve 10 GPU saatiydi). Haftalık yük ortalama 12 saattir. Sığmazsa demo cilası ya da kural kartı sayısı küçültülür; bu, Kapı B'de kararlaştırılır.

Takvim varsayımı: 1. hafta 16 Kasım 2026 Pazartesi başlar, 8. hafta 10 Ocak 2027'de biter. Teslim tarihleri netleşince kaydırılır.

## İş yükü

| İş paketi | Hafta | Aktif çalışma, YZ destekli (saat) | GPU (saat) | Çıktı |
| --- | --- | --- | --- | --- |
| 1. Kurulum ve veri | H1 | 10 | 0 | Ortak biçimde üç test seti, iki eğitim alt kümesi, veri raporu |
| 2. Değerlendirme hattı | H2 | 10 | 1 | Kapı A: mT5'in 79,0 skoru yeniden üretildi |
| 3. Model seçimi ve pilot | H3 | 12 | 3 ile 4 | Kapı B: seçilen model, ölçülmüş hız ve `max_len`, keşif fazının teknik listesi |
| 4. İki eğitim (D1, D2) | H4 ile H5 | 8 | 5 ile 11 | Üç sette F0.5 sonuç tablosu |
| 4b. Keşif fazı (K25) | H5 ile H6 | 10 ile 15 | 7 ile 12 | Doğrulama setinde teknik kıyası, nihai yapılandırma, boyut merdiveni |
| 5. Analiz ve kural kartları | H6 | 12 ile 18 | 1 | Kural bazında analiz, güven aralıkları, verimlilik tablosu, görülmemiş alan ölçümleri, 25 kural kartı |
| 6. Demo, rapor, sunum | H7 ile H8 | 22 | 0 | Gradio demosu, rapor, sunum |
| **Toplam** | **8 hafta** | **~90–100** | **~20–25** | |

Rakamlar kaba tahmindir. D2, D1'in iki katı veri gördüğü için GPU süresinin çoğu oradadır. Kesin süre 3. haftadaki pilotta ölçülür. Yapay zeka en çok kod, betik, demo ve rapor taslağını hızlandırır; GPU süresini, hata ayıklamayı ve savunmada anlatılacak şeyleri anlama süresini kısaltmaz.

## Paketler

### 1. Kurulum ve veri (H1)

Görev dosyası: `gorevler/01_veri_hazirligi.md`

- [x] Danışmanla sadeleştirilmiş kapsamı onayla (2026-11-21: danışman uygun gördü; model sayısının artırılabileceğini söyledi)
- [x] Depoyu kur, Claude Code'u çalıştır, testleri geçir
- [x] GECTurk, OSCAR GEC (eğitim ve test) ve #Turki$hTweets'i indir; şemalarını incele (GECTurk M2 dosyaları 2026-11-24)
- [x] Lisansları ve Hugging Face kimliklerini doğrula (`raporlar/01_veri_kaynaklari.md`)
- [x] Hepsini ortak JSONL biçimine çevir; özgün M2 dosyalarını skorlama için sakla (2026-11-24: GECTurk'ün dört dosyası da çevrildi; düzenlemeler hedef dosyalarını birebir üretiyor)
- [x] Sızıntı taraması (bütün eğitim ve test çiftleri) (2026-11-24: bütün çiftlerde 0; GECTurk train'den dev ve test ile çakışan 184 cümle çıkarıldı)
- [x] OSCAR'dan 500 cümlelik doğrulama dilimi ayır
- [x] Katmanlı örneklemeyle D1 ve D2 alt kümelerini üret (GECTurk 12 bin: 6.000 hatalı + 6.000 hatasız, her kuraldan en az 50; OSCAR 12 bin: 9.000 hatalı + 3.000 hatasız, K27)
- [ ] Her setten 20 örneği elle kontrol et (komut hazır: `scripts/show_examples.py`; kullanıcı bakacak)

### 2. Değerlendirme hattı (H2)

Görev dosyası: `gorevler/02_degerlendirme_hatti.md`

- [x] Ersoy ve Yıldız'ın hangi skorlayıcıyı kullandığını doğrula (ERRANT-TR; K15 notu)
- [x] ERRANT-TR'nin kullanılabilir olup olmadığını doğrula (kullanılabilir, MIT)
- [x] ERRANT-TR'yi kur; OSCAR, tweetler ve film yorumlarında skorlayıcı olarak bağla (2026-11-24: üç sette de çalışıyor)
- [x] mT5 modelini üç sette çalıştır; tweetlerde 79,0'ı yeniden üret (tweetler 78,5 / 79,0; OSCAR testi 55,1 / 55,4; film yorumları 28,8 / 28,1; Kapı A geçildi)
- [ ] M2 scorer'ı film yorumları için kur, zaman aşımı ekle
- [ ] GECTurk etiketleyicisini çalıştırmayı dene (en fazla yarım gün)
- [x] Skorlayıcıdan cümle bazlı sayıları çıkar, bootstrap'a bağla (ERRANT-TR için; M2 scorer bekliyor)

**Kapı A:** mT5'in tweetlerdeki skoru ±2 puan içinde tutuyor. Tutmuyorsa eğitime geçilmez, hat düzeltilir.

### 3. Model seçimi ve pilot (H3)

Görev dosyası: `gorevler/03_pilot_ve_model_secimi.md`

- [x] Aday modellerin Hugging Face kimliklerini ve lisanslarını doğrula (`raporlar/03_model_adaylari.md`, bölüm 9)
- [x] Eğitim örneklerini iki talimatlı biçime çevir (`trgec.examples`; 2026-11-24)
- [ ] Token uzunluklarını ölç, `max_len` değerini belirle (2026-11-24: gerçek örneklerle ölçüldü, 512 tavanı 24.000 örnekten en fazla 10'unu eliyor; nihai değer Kapı B'de)
- [x] `train.py` ve `predict.py` yaz; yerelde duman testini geçir (2026-11-24; ufakzeka-1 ile CPU'da)
- [ ] Sekiz adayı 2 bin örnekle mini eğitimden geçir, doğrulama setinde kıyasla (K7 notu)
- [ ] Boyut merdiveninin iki küçük basamağını ve keşif fazının teknik listesini belirle (K25)
- [ ] Tepe GPU belleğini 512 token tavanıyla ölç (`raporlar/03_model_adaylari.md`, bölüm 10)
- [ ] Adım başına hızı ölç, D1 ve D2 için süre tahmini çıkar
- [ ] fp16'da NaN kayıp olup olmadığını izle

**Kapı B:** Ana model, veri boyutu, `max_len`, değerlendirme protokolü ve keşif fazının teknik listesi seçildi; bundan sonra değişmez (K20 notu). B planı ölçütleri (K21) burada kontrol edilir.

### 4. İki eğitim (H4 ile H5)

- [ ] D1 eğitimi; adaptör Hugging Face Hub'a
- [ ] D2 eğitimi; adaptör Hugging Face Hub'a
- [ ] Emniyet kemeri eşiğini doğrulama setinde ayarla
- [ ] İki modelin üç test setindeki tahminlerini üret
- [ ] Sonuç tablosunu doldur

GPU çalışırken 5. paketin betikleri hazırlanır; beklemeye gerek yoktur.

### 4b. Keşif fazı (H5 ile H6)

Karar: K25. Bütün denemeler yalnızca doğrulama setlerinde değerlendirilir.

- [ ] Sentetik gürültü ekleme: gürültü üreticisini yaz ve test et, karışımla eğit
- [ ] Boyut merdiveni: iki küçük modeli D2 ile eğit
- [ ] Nicemleme kıyası (fp16, 8-bit, 4-bit) ya da GPT GEC verisiyle üçüncü karışım
- [ ] Doğrulama setinde en iyi yapılandırmayı seç, günlüğe yaz ve dondur
- [ ] Nihai yapılandırmayı bir kez eğit; test setlerinde bir kez ölç

### 5. Analiz ve kural kartları (H6)

- [ ] Kural bazında recall (film yorumları)
- [ ] Hata türü bazında analiz (OSCAR ve tweetler)
- [ ] Emniyet kemeri açık ve kapalıyken aşırı düzeltme oranı
- [ ] Model çiftleri için bootstrap güven aralıkları
- [ ] Bozuk çıktı oranı
- [ ] Verimlilik tablosu: bizim modeller ve iki temel model (K26)
- [ ] Görülmemiş alan: hatasız metinde yanlış düzeltme oranı ve BOUN de/da testi (K26)
- [ ] 25 kural kartını TDK sayfalarından taslakla, elle doğrula
- [ ] 15 ile 20 başarısız örneği niteliksel olarak incele

### 6. Demo, rapor, sunum (H7 ile H8)

- [ ] Gradio demosu: hataların altı çizili, tıklayınca kural kartı açılıyor
- [ ] Hugging Face Spaces'te yayınla (ücretsiz katman CPU; gerekirse 4-bit GGUF)
- [ ] Rapor: giriş, literatür, veri, yöntem, sonuçlar, sınırlılıklar
- [ ] Figürler: üç sette model karşılaştırması, kural bazında recall
- [ ] Sunum ve canlı demo provası

## Yedek liste: zaman kalırsa

Çekirdekten çıkarılan işler. Çekirdek erken biterse yukarıdan aşağıya eklenir. Saatler kaba tahmindir. 2026-11-23'te iki madde plana alındı: BOUN de/da testi (K26) ve ikinci küçük modelin eğitimi (boyut merdiveni, K25).

| İş | Ne katar | Ek aktif saat | Ek GPU saati |
| --- | --- | --- | --- |
| Nihai modeli 3 tohumla tekrarlamak | Sonuçların tesadüf olmadığını gösterir | 3 | 10 |
| Büyük bir ticari LLM'i yalnızca 300 cümlede denemek | Üst sınırı gösterir | 4 | 0 (API ücreti) |
| D2 için eşit boyutlu kontrol (6 bin + 6 bin) | "Daha çok veri mi, organik veri mi?" sorusunu ayırır | 2 | 3 |
| GPT GEC verisiyle üçüncü karışım (D3) | Veri sorusunu genişletir | 5 | 5 |
| 200 ile 300 cümlelik kendi test setimizi etiketlemek | Özgün veri katkısı, istatistiksel güç | 15 ile 20 | 0 |
| Serbest metin açıklama damıtma ve insan değerlendirmesi | Kural kartının ötesinde doğal açıklama | 20 | 12 |
