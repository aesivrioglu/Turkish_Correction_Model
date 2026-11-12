# Teknik notlar ve tuzaklar

Bu belge planın teknik incelemesinde bulunan sorunları, doğrulanmamış varsayımları ve bilinen tuzakları toplar. Her madde bir görevle eşleşir; görev tamamlanınca maddenin altına sonuç yazılır.

## Planın incelemesinde bulunan yedi sorun

| No | Sorun | Neden sorun | Çözüm | Görev | Karar |
| --- | --- | --- | --- | --- | --- |
| 1 | 128 token sınırı fazla kısa | Talimat, cümle, düzeltilmiş cümle ve düzeltme listesi sığmaz; hedefler kesilir | Uzunlukları ölç, %95'lik dilime göre sınır koy; JSON yerine kısa satır biçimi | 03 | K8, K11 |
| 2 | OSCAR örneklerinde kural etiketi yok | Hepsine `OTHER_SPELL` yazmak modelin etiketlemeyi yanlış öğrenmesine yol açar | İki talimat: GECTurk için "düzelt ve etiketle", OSCAR için "yalnızca düzelt" | 03 | K9 |
| 3 | Rastgele alt küme nadir kuralları siler | 25 kuralın sıklığı çok dengesiz | Kural başına alt sınırlı katmanlı örnekleme | 01 | K10 |
| 4 | Skorlayıcı tutarsızlığı | Farklı araçlar farklı skor verir; yayımlanmış sayılarla kıyas geçersiz olur | Her sette yayımlanmış skorun aracını kullan | 02 | K15 |
| 5 | Test setinde ayar yapma riski | Eşik ve model seçimi için gerçek alandan doğrulama seti yok | OSCAR eğitim verisinden 500 cümlelik doğrulama dilimi | 01 | K16 |
| 6 | Etiketleyici çalıştırılmazsa anlamlılık testi düşer | Eşleştirilmiş bootstrap cümle bazlı çıktı ister | Yarım günle sınırlı deneme; olmazsa sınırlılık olarak yaz | 02 | K17 |
| 7 | Sıfır-atışla model seçimi yanıltıcı | Eğitimsiz başarı, eğitim sonrasını öngörmez | 2 bin örnekle mini eğitim | 03 | K7 |

## Doğrulanmamış varsayımlar

Bunlar planda doğru kabul edildi ama kontrol edilmedi. İlgili görevde ilk iş olarak doğrulanır.

| Varsayım | Yanlışsa ne olur | Nasıl doğrulanır | Görev |
| --- | --- | --- | --- |
| ERRANT-TR kullanılabilir bir araç olarak yayımlanmış | OSCAR ve tweetlerde hata türü analizi yapılamaz | Makalenin ve yazarların depolarını ara. Yoksa tweet setinin kendi 13 etiketini kullan | 02 |
| OSCAR GEC ve tweet setinin lisansı akademik kullanıma izin veriyor | Veri kullanılamaz ya da paylaşılamaz | Depo ve veri kartlarındaki lisansı oku, `docs/raporlar/` altına yaz | 01 |
| OSCAR GEC eğitim ve test verisi, mT5 modeli Hugging Face'te erişilebilir | Veri ya da temel model elde edilemez | github.com/asimokby/Turkish-GEC README'sindeki bağlantıları aç | 01 |
| Ersoy ve Yıldız'ın skorları M2 scorer ile hesaplanmış | Yanlış araçla 79,0 yeniden üretilemez | Makalenin değerlendirme bölümünü ve depodaki betikleri oku | 02 |
| Aday modeller T4'te fp16 ile kararlı eğitilir | Kayıp NaN olur | Mini eğitimde kaybı izle | 03 |
| Kumru-2B ve Qwen küçük modelinin Hugging Face kimlikleri `configs/base.yaml` içindeki gibi | Model indirilemez | Model kartlarını aç, lisansı ve sohbet şablonunu kontrol et | 03 |

**Sonuç (2026-11-12, `raporlar/00_varsayim_dogrulama.md`):**

- ERRANT-TR yayımlanmış (MIT) ve kullanılabilir.
- OSCAR GEC verisinin lisansı MIT. Tweet seti ve mT5 ağırlıklarının lisansı yok.
- OSCAR GEC verisi ve mT5 modeli Hugging Face'te var.
- Ersoy ve Yıldız M2 scorer **değil**, ERRANT-TR kullanmış.
- fp16 kararlılığı pilotta ölçülecek.
- Kumru'nun kimliği `vngrs/Kumru-2B` olarak düzeltildi.
- Yeni tuzak: GECTurk'ün HF kopyasında düzeltilmiş cümle yok; M2 dosyaları Drive'da.

## Bilinen tuzaklar

### Türkçe büyük ve küçük harf

Python'da `"I".lower()` sonucu `"i"`, `"İ".lower()` sonucu ise `"i"` artı birleşen nokta olur. İkisi de Türkçe için yanlıştır. Karşılaştırma ve yineleme taramasında `trgec.text.tr_lower` kullanılmalı. Eğitim verisine küçük harfe çevirme uygulanmaz; büyük harf hataları GECTurk kurallarının parçasıdır.

Bu tuzak gerçek veride de karşımıza çıktı. #Turki$hTweets'in kaynak sütununda 47 birleşen nokta (U+0307) var (`yedi̇m`); veri seti `str.lower()` ile küçültülmüş. `normalize_for_match` bu noktayı noktalama sayıp kelimeyi bölüyordu (`yedi m`). 2026-11-13'te düzeltildi ve test eklendi. Ölçütün kaynak metni değiştirilmez.

### Noktalama ve boşluklar

M2 biçimi boşlukla ayrılmış token dizini kullanır. GECTurk cümlelerinde noktalamanın ayrı token olup olmadığı görev 01'de incelenmeli. Model noktalama çevresindeki boşlukları "düzeltirse" skorlayıcı bunu hata sayar. Eğitim hedefleri kaynakla aynı tokenizasyonu korumalı.

Görev 01 sonucu (`raporlar/01_tokenizasyon.md`):

- GECTurk'te noktalama kelimeye bitişik.
- OSCAR testinin `inp`/`ref` dosyaları doğal metin; M2 dosyasında noktalama ayrı, çünkü ERRANT-TR Zemberek ile kendisi tokenize ediyor.
- Tweetlerde `|` işareti boşluk anlamına geliyor.
- Ortak biçimde doğal metin saklanması önerildi ve kullanıcı onayı bekliyor.

### Sızıntı

OSCAR bir web taramasıdır. Film yorumları (Beyazperde) ve tweetler OSCAR eğitim verisinde bulunabilir. Yalnızca OSCAR eğitim ile OSCAR test değil, bütün eğitim ve test çiftleri taranır.

### M2 scorer yavaşlığı

M2 scorer, kaynak cümleden çok farklı hipotezlerde aşırı yavaşlayabilir. Cümle başına zaman aşımı konur. Emniyet kemeri bu durumu azaltır ama kemer kapalı ölçümde sorun çıkabilir.

### Kayıp yalnızca cevap tokenlarında

Eğitimde talimat ve kaynak cümle tokenları kayıptan çıkarılır (etiket -100). Yoksa model talimatı ezberlemeye çalışır.

### Üretim ayarları

Çıkarımda açgözlü kod çözme kullanılır (örnekleme kapalı). `max_new_tokens`, kaynak cümlenin uzunluğuna göre sınırlanır; model durmazsa çıktı kesilir ve bozuk sayılır.

### Sohbet şablonu

Her modelin sohbet şablonu farklıdır. Elle metin birleştirmek yerine tokenizer'ın `apply_chat_template` işlevi kullanılır. Eğitim ve çıkarım aynı şablonu kullanmalı.

### Kaggle oturumları

Etkileşimli oturum boştayken de kota harcar. Eğitim "Save Version" ile toplu modda çalıştırılır. Oturum en fazla 12 saat sürer; adaptör eğitim bitince hemen Hugging Face Hub'a yüklenir, çünkü oturum kapanınca disk silinir.

### Hatasız cümleler

GECTurk'ün yarısı hatasız cümledir. Alt kümede bu oran korunur. Hatasız örnekler olmadan model her cümlede hata arar ve aşırı düzeltme artar.
