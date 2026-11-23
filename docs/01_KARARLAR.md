# Kararlar ve nedenleri

Bu belge projede alınan kararları, her birinin nedenini ve neye mal olduğunu kaydeder. Bir karar değişirse eski madde silinmez; altına tarihli bir not düşülür.

Her madde üç bölümden oluşur: **Karar**, **Neden**, **Bedel** (bu kararla vazgeçtiğimiz şey).

## Kapsam kararları

### K1. Kapsam çekirdeğe indirildi

- **Karar:** İlk plandaki dört model ailesi, beş deney grubu ve açıklama damıtma yerine tek model, iki eğitim ve üç test seti.
- **Neden:** İlk plan yaklaşık 26 hafta ve 85 GPU saati tutuyordu. Aynı dönemde iki proje daha yürüyor. Her şeye biraz dokunup hiçbirini bitirememek en büyük riskti.
- **Bedel:** Bulgunun başka modellerde de geçerli olup olmadığını göstermiyoruz. Çıkarılan işler `02_IS_PLANI.md` içindeki yedek listede duruyor.

**2026-11-23 notu (kullanıcı onayıyla):** Danışman projeyi uygun gördü ve model sayısının artırılabileceğini söyledi; kullanıcı da hedefi "hafif modellerle, iyi genellenen bir düzeltici" olarak netleştirdi. Çekirdek korunur: seçilen model, D1 ve D2, üç test seti. Eklenenler:

- Pilotta iki yerine sekiz aday (K7 notu).
- Yalnızca doğrulama setlerinde çalışan, en fazla üç teknikle sınırlı bir keşif fazı (K25).
- Verimlilik ölçümleri ve görülmemiş alanlarda ek ölçümler (K26).

Yeni bedel: aktif çalışma tahmini 72 saatten 90–100 saate, GPU 10 saatten 20–25 saate çıkar (kaba tahmin). 8 haftaya sığması için demo cilası ya da kural kartı sayısı küçülebilir; bu, Kapı B'de ölçülen hıza göre kararlaştırılır.

### K2. Genişlik kesildi, değerlendirme derinliği korundu

- **Karar:** Model ve varyant sayısı azaltıldı; üç gerçek test seti, kural bazında analiz ve güven aralıkları kaldı.
- **Neden:** Projeyi "küçük bir modeli QLoRA ile eğittim, demo yaptım" seviyesine indirmek Text Mining projesinden farkını yok ederdi. Ayırt edici olan titiz değerlendirme.
- **Bedel:** Değerlendirme hattı projenin teknik olarak en zor kısmı olarak kalıyor.

### K3. Araştırma soruları ikiye indirildi

- **Karar:** Yalnızca AS1 (performans) ve AS2 (veri karışımı). Açıklama damıtma sorusu çıkarıldı, aşırı düzeltme bir metrik oldu.
- **Neden:** Bir soruyu temiz cevaplamak, dört soruyu yarım cevaplamaktan daha güçlü bir capstone çıkarır.
- **Bedel:** "Açıklama üretmek düzeltme başarısını etkiler mi?" sorusu cevapsız kalıyor.

**2026-11-23 notu (kullanıcı onayıyla):** Soru sayısı iki olarak kalır, kapsamları genişler:

- **AS1, performans ve verimlilik:** Ücretsiz bir T4'te 4-bit nicemleme ve LoRA ile ince ayarlanan düşük parametreli açık modeller (yaklaşık 0,2–2,4B) üç gerçek test setinde yayımlanmış sonuçlara ne kadar yaklaşıyor ya da onları geçiyor; model küçüldükçe ne kaybediliyor?
- **AS2, veri ve genellenebilirlik:** Sentetik veriye organik yazım hataları ve keşif fazındaki teknikleri eklemek, eğitimde görülmemiş alanlara geçişi nasıl etkiliyor; hangi kurallarda yine de yetmiyor?

"Hafif" sözü büyük LLM'lere göredir. Yayımlanmış iki temel model bizim ana adaylardan küçüktür (etiketleyici yaklaşık 110M, mT5 582M); "önceki modellerden hafif" iddiası yoktur. Üç test seti hedef değil, genellenebilirliğin ölçüsüdür.

### K4. Üç gerçek test seti kullanılacak

- **Karar:** GECTurk film yorumları (300 cümle), OSCAR GEC test (2.400 cümle), #Turki$hTweets (2.000 tweet).
- **Neden:** GECTurk'ün kendi sentetik testi doymuş durumda (etiketleyici F0.5 = 0,98), modelleri ayırt edemiyor. Tek gerçek set olan film yorumları 300 cümle ve istatistiksel gücü düşük.
- **Bedel:** Üç farklı biçim ve etiket şemasını tek hatta toplamak gerekiyor.

## Model ve eğitim kararları

### K5. Tek model, iki eğitim (D1 ve D2)

- **Karar:** D1 yalnızca GECTurk alt kümesiyle, D2 aynı GECTurk alt kümesi artı OSCAR GEC alt kümesiyle eğitilir.
- **Neden:** AS2'nin cevabı bu iki eğitimin farkından çıkar. GECTurk kısmı iki eğitimde birebir aynı olduğu için fark OSCAR verisine bağlanabilir.
- **Bedel:** D2 iki kat veri görür; "organik veri mi yardım etti, yoksa daha fazla veri mi?" sorusu tam ayrışmaz. Raporda sınırlılık olarak yazılır. Eşit boyutlu kontrol deneyi yedek listede.

**2026-11-23 notu (kullanıcı onayıyla):** D1 ve D2 ana kontrollü karşılaştırma olarak aynen kalır. Bunlara ek eğitimler yalnızca keşif fazı (K25) kapsamında yapılır.

### K6. Eğitim küçültüldü

- **Karar:** 1 ile 2 milyar parametre arası model, GECTurk'ten 12 bin ve OSCAR'dan 12 bin cümle, 1 epok.
- **Neden:** Uzun eğitimler riskli: oturum 12 saatte kesilir, bir hata saatlerce emeği siler. GECTurk makalesinde eğitim verisinin dörtte biriyle bile yüksek skor alınıyor, yani tüm veri şart değil.
- **Bedel:** Daha büyük model ya da daha çok veriyle alınabilecek birkaç puan. Sayılar pilotta ölçülen hıza göre güncellenir.

**2026-11-23 notu (kullanıcı onayıyla):** Boyut aralığı "1 ile 2 milyar" yerine "yaklaşık 0,2 ile 2,4 milyar" oldu (sekiz adayın en küçüğü 0,18B, en büyüğü 2,38B). Veri boyutu ve epok sayısı değişmedi. Tahmini süreler `raporlar/03_model_adaylari.md` içinde.

### K7. Model seçimi mini eğitimle yapılır, sıfır-atışla değil

- **Karar:** İki aday model 2 bin örnekle kısa birer eğitimden geçirilir ve doğrulama setinde kıyaslanır.
- **Neden:** Küçük modellerin eğitimsiz başarısı eğitim sonrasını iyi öngörmez. Sıfır-atışta kötü görünen model ince ayardan sonra daha iyi olabilir.
- **Bedel:** Yaklaşık 1 GPU saati ek maliyet.

**2026-11-23 notu (kullanıcı onayıyla):** Mini eğitimden iki yerine sekiz aday geçer (`raporlar/03_model_adaylari.md`, bölüm 9): `vngrs/Kumru-2B`, `newmindai/Mecellem-Qwen3-1.7B-TR`, `Qwen/Qwen3-1.7B`, `utter-project/EuroLLM-1.7B-Instruct`, `asafaya/kanarya-2b`, `Qwen/Qwen3-0.6B`, `ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1`, `ufakai/ufakzeka-1`. Ana deney yine tek modelle yapılır; pilot ayrıca boyut merdiveninin küçük basamaklarını seçer (K25). Yeni bedel: yaklaşık 1 yerine 3–3,5 GPU saati (tahmin).

### K8. Çıktı JSON değil, kısa satır biçimi

- **Karar:** Düzeltilmiş cümle, `###` satırı, ardından her satırda `yanlış => doğru | KURAL`.
- **Neden:** JSON aynı bilgiyi daha çok token ile taşır ve küçük modeller geçersiz JSON üretebilir. Kısa biçim hem eğitimi hızlandırır hem bozuk çıktı oranını düşürür.
- **Bedel:** Kendi ayrıştırıcımızı yazdık (`trgec.formats.parse_output`); bozuk çıktı oranı ayrıca raporlanmalı.

### K9. İki ayrı talimat

- **Karar:** GECTurk örnekleri "düzelt ve etiketle", OSCAR örnekleri "yalnızca düzelt" talimatıyla eğitilir. Çıkarımda "düzelt ve etiketle" kullanılır.
- **Neden:** OSCAR verisinde kural etiketi yok. Hepsine `OTHER_SPELL` yazsaydık, OSCAR'daki de/da gibi hatalar da bu etiketi alır ve model etiketlemeyi yanlış öğrenirdi.
- **Bedel:** Model iki görevi birlikte öğrenmek zorunda. GECTurk dışı hataların etiketi çıkarımda belirsiz kalabilir; bunlar demoda "genel yazım hatası" olarak gösterilir.

### K10. Katmanlı örnekleme

- **Karar:** GECTurk alt kümesi rastgele değil, kural başına en az 50 örnek garantisiyle seçilir (`trgec.sampling.stratified_sample`).
- **Neden:** 25 kuralın sıklığı çok dengesiz. Rastgele 12 bin cümlelik örneklemde nadir kurallardan birkaç örnek kalır ve model o kuralları öğrenemez.
- **Bedel:** Alt kümedeki kural dağılımı gerçek dağılımdan biraz sapar.

### K11. Token sınırı ölçülerek konur

- **Karar:** `max_len` tahminle değil, seçilen modelin tokenizer'ıyla ölçülen uzunlukların %95'lik dilimine göre belirlenir (`trgec.lengths.suggest_max_len`).
- **Neden:** Daha önce önerilen 128 token; talimat, cümle, düzeltilmiş cümle ve düzeltme listesi için büyük olasılıkla yetmez. Kesilen hedefler modele yarım cevap vermeyi öğretir.
- **Bedel:** Sınır büyüdükçe eğitim yavaşlar.

**2026-11-23 notu (kullanıcı onayıyla):** Ölçüm yapıldı (`raporlar/03_model_adaylari.md`, bölüm 10) ve üç kural eklendi:

- OSCAR alt kümesi ve doğrulama dilimi, kaynak cümlesi en fazla 70 kelime olan satırlardan seçilir. Süzme kelime sayısıyla yapılır ki bütün adaylar aynı veriyi görsün.
- Pilotta bütün adaylar için token tavanı 512'dir. Tavanı aşan örnek kırpılmaz, o model için eğitimden çıkarılır.
- Nihai `max_len` yine Kapı B'de, seçilen modelin tokenizer'ı ve gerçek alt kümelerle bu kararın yöntemiyle (%95'lik dilim) ölçülür; 512'yi geçemez.

Yeni bedel: model eğitimde 70 kelimeden uzun OSCAR satırı görmez; çıkarımda uzun girdilerin cümlelere bölünmesi gerekir.

### K12. Text Mining projesinden farklı model ailesi

- **Karar:** Ana model Gemma ailesinden seçilmez. Adaylar Kumru-2B ve Qwen ailesinden küçük bir model.
- **Neden:** Text Mining projesinde Gemma 4 E2B ile QLoRA kullanılıyor. Farklı aile, iki projenin ayrışmasını görünür kılar. Ayrıca fp16 kararsızlığı riski olan modellerden kaçınılır.
- **Bedel:** Text Mining'deki model deneyimi doğrudan taşınmaz; eğitim kodu taşınır.

## Açıklama ve güvenlik kararları

### K13. Açıklama kural kartından gelir

- **Karar:** Model yalnızca kural etiketini üretir. Kullanıcıya gösterilen açıklama, TDK yazım kurallarına dayanan 25 sabit kural kartından alınır.
- **Neden:** Model açıklama metni üretirse kural uydurabilir. Sabit kart hem güvenilir hem ucuz; damıtma ve insan değerlendirmesi gerektirmez.
- **Bedel:** Açıklamalar cümleye özel değil, kurala özeldir.

### K14. Emniyet kemeri

- **Karar:** Model çıktısı kaynak cümleyi belirli bir orandan fazla değiştirdiyse orijinal cümle korunur (`trgec.guard.apply_guard`).
- **Neden:** LLM'ler düzeltmek yerine yeniden yazmaya eğilimlidir. GEC'te yanlış düzeltme, kaçırılan hatadan daha zararlıdır; F0.5 de bu yüzden precision ağırlıklıdır.
- **Bedel:** Çok hatalı cümlelerde doğru düzeltmeler de geri alınabilir. Kemer açık ve kapalıyken skorlar ayrı raporlanır.

## Değerlendirme kararları

### K15. Her sette yayımlanmış skorun aracı kullanılır

- **Karar:** Film yorumlarında M2 scorer. OSCAR ve tweet setlerinde, Ersoy ve Yıldız'ın skorlarını ürettiği değerlendirme betiği.
- **Neden:** Farklı skorlayıcılar aynı çıktıya farklı skor verir. Yayımlanmış sayılarla kıyas ancak aynı araçla anlamlıdır.
- **Bedel:** İki ayrı değerlendirme yolu kurmak gerekebilir. Hangi aracın kullanıldığı görev 02'de doğrulanacak.

**2026-11-13 notu (kullanıcı onayıyla):** Araştırma, Ersoy ve Yıldız'ın üç test setinin hepsini ERRANT-TR ile skorladığını gösterdi (makale, bölüm 4.2; ayrıntı `raporlar/00_varsayim_dogrulama.md`). Film yorumlarındaki iki yayımlanmış sayı farklı araçlardan geliyor: etiketleyicinin 84,2'si GECTurk'ün M2 scorer'ıyla, mT5'in 28,1'i ERRANT-TR ile hesaplanmış. Karar şöyle güncellendi:

- OSCAR GEC test ve tweetler: ERRANT-TR (`github.com/harunuz/erranttr`, `commands/compare_m2.py`, F0.5).
- Film yorumları: iki araç birlikte çalıştırılır. M2 scorer sonucu 84,2 ile, ERRANT-TR sonucu 28,1 ile kıyaslanır. Raporda her sayının yanına aracı yazılır.
- Yeni bedel: ERRANT-TR iki yeni bağımlılık ister (`rapidfuzz`, `zemberek-python`); eklenmeden önce onay alınır. Yazarlar tam komutu yayımlamadığı için bayraklar Kapı A'da 79,0 yeniden üretilerek sabitlenir.

**2026-11-24 notu (kullanıcı seçimi bana bıraktı):** Kapı A geçildi (tweetlerde 78,5; OSCAR testinde 55,1). ERRANT-TR'nin iki karşılaştırma modu (`cs`: konum ve metin; `cse`: ayrıca hata türü) yayımlanmış skorları aynı ölçüde üretiyor; aralarındaki fark tweetlerde 1, OSCAR testinde 2 düzenleme. **Ana skor `cs`** (`eval.errant_tr_mode`): ERRANT'ın varsayılanı ve yazındaki standart ölçü; ana skoru etiket eşlemesinin ayrıntısına bağlamıyor. `cse` her koşuda hesaplanıp yanında raporlanır. Ayrıntı `raporlar/02_skorlayici.md`.

### K16. Test setlerine yalnızca en sonda bakılır

- **Karar:** Model seçimi, emniyet kemeri eşiği ve diğer ayarlar doğrulama setlerinde yapılır: GECTurk dev ve OSCAR eğitim verisinden ayrılan 500 cümlelik dilim.
- **Neden:** Test setinde ayar yapmak skoru şişirir ve sonucu geçersiz kılar. Gerçek alandan hazır bir doğrulama seti olmadığı için OSCAR'dan dilim ayrılır.
- **Bedel:** Film yorumu ve tweet alanı için doğrulama verisi yok; eşik bu alanlara tam uymayabilir.

### K17. Etiketleyici zaman sınırlı olarak yeniden çalıştırılır

- **Karar:** GECTurk etiketleyicisini çalıştırmak için en fazla yarım gün harcanır. Çalışmazsa yayımlanmış 84,2 skoru referans alınır.
- **Neden:** Eşleştirilmiş bootstrap için etiketleyicinin cümle bazlı çıktıları gerekir; yalnızca yayımlanmış skorla anlamlılık testi yapılamaz. Öte yandan 2023 deposunun bağımlılıkları sorun çıkarabilir ve bu iş projeyi kilitlememeli.
- **Bedel:** Çalışmazsa yalnızca kendi skorumuzun güven aralığı verilir ve bu, raporda sınırlılık olarak yazılır.

### K18. Güven aralıkları zorunlu

- **Karar:** Her F0.5 skoru bootstrap güven aralığıyla raporlanır; iki sistem arasındaki fark yalnızca aralık sıfırı içermiyorsa anlamlı sayılır.
- **Neden:** 300 cümlelik sette birkaç puanlık fark tesadüf olabilir.
- **Bedel:** Skorlayıcıdan cümle bazlı sayıları çıkarmak gerekiyor.

## Süreç kararları

### K19. Kapı A: skorlar tutmadan eğitim yok

- **Karar:** Yayımlanmış en az bir temel skor (mT5'in tweetlerdeki 79,0'ı) ±2 puan içinde yeniden üretilmeden model eğitilmez.
- **Neden:** Değerlendirme hattındaki bir hata (tokenizasyon, biçim) sonraki bütün sonuçları geçersiz kılar. Bunu eğitimden önce yakalamak en ucuz yoldur.
- **Bedel:** İkinci hafta yalnızca altyapıya gider.

### K20. Kapı B: model ve veri boyutu donar

- **Karar:** Üçüncü haftanın sonunda seçilen model, veri boyutu ve `max_len` sonraki deneylerde değişmez.
- **Neden:** Deneyler arasında birden çok şey değişirse farkın nedeni söylenemez.
- **Bedel:** Sonradan daha iyi bir ayar bulunsa bile kullanılmaz; yedek listeye yazılır.

**2026-11-23 notu (kullanıcı onayıyla):** Kapı B'nin kapsamı keşif fazına (K25) göre yeniden tanımlandı. Kapı B'de donanlar:

- ana model, D1 ve D2 alt kümeleri, `max_len`;
- değerlendirme protokolü (skorlayıcılar, dönüşümler, emniyet kemeri ayar yöntemi);
- keşif fazında denenecek tekniklerin listesi (en fazla üç).

Keşif fazı yalnızca doğrulama setlerinde yürür. Nihai yapılandırma test setleri açılmadan önce donar ve günlüğe yazılır. Her yapılandırma test setlerinde bir kez ölçülür; test sonucuna bakıp ayar değiştirilmez.

### K21. B planı: eğitimsiz değerlendirme çalışması

- **Karar:** Üçüncü haftanın sonunda şunlardan biri geçerliyse proje eğitimsiz bir değerlendirme çalışmasına çevrilir: (a) D2 eğitimi tek oturuma sığmıyor ve veri küçültülünce doğrulama skoru çöküyor, (b) fp16 kararsızlığı iki adayda da çözülemiyor, (c) mini eğitimli model doğrulama setinde eğitimsiz halinden iyi değil.
- **Neden:** İlk üç haftanın işi (veri, değerlendirme hattı, pilot) iki planda da aynıdır; geçiş kayıpsızdır. Eğitim riski projeyi batırmamalı.
- **Bedel:** B planında "kendi modelini eğittim" katkısı kaybolur ve soru Text Mining projesine yaklaşır. Bu yüzden B planı ilk tercih değildir.

### K22. Hibrit ortam: yerelde Claude Code, GPU için Kaggle

- **Karar:** Kod yerelde yazılır ve test edilir; Kaggle not defteri yalnızca depoyu klonlayıp betik çalıştırır.
- **Neden:** Claude Code Kaggle'ın tarayıcı editöründe çalışmaz. Not defterlerindeki gizli durum (hücre sırası, eski değişkenler) hataya açıktır. İşin büyük kısmı zaten GPU istemez.
- **Bedel:** GPU'daki hatalar doğrudan görünmez; günlükleri geri taşımak gerekir. Ayrıntı `04_ORTAM_VE_IS_AKISI.md` içinde.

### K23. QLoRA katkı değil, araçtır

- **Karar:** Raporda ve sunumda QLoRA, karşılaştırılan üç sistemden birini eğitmenin aracı olarak anlatılır.
- **Neden:** Text Mining projesiyle örtüşme yalnızca araç düzeyinde. Görev, veri, değerlendirme ve araştırma sorusu farklı.
- **Bedel:** Yok.

## Veri biçimi kararları

### K24. Ortak biçimde doğal metin saklanır

Tarih: 2026-11-21 (kullanıcı onayıyla). Ayrıntı: `raporlar/01_tokenizasyon.md`.

- **Karar:** `data/processed/*.jsonl` dosyalarında `source` ve `target` doğal metindir; hiçbir sete tokenizasyon uygulanmaz. Her skorlayıcı kendi tokenizasyonunu yapar. Ölçüte özgü dönüşümler (tweetlerde altın cümlenin ilk harfini büyütme, model çıktısından noktalama silme) JSONL'de değil, değerlendirme katmanında uygulanır. Tweetlerdeki `|` işareti boşluğa çevrilir. Özgün etiketler `extra` alanında saklanır.
- **Neden:** Model gerçek kullanımda ve demoda doğal metin görecek. GECTurk'ün doğal metni zaten M2 scorer'ın beklediği biçimde (boşlukla bölünür); ERRANT-TR de girdiyi Zemberek ile kendisi tokenize ediyor. Dönüşümler tek yerde durunca hangi sayının hangi dönüşümle üretildiği açık kalır.
- **Bedel:** Model noktalama çevresindeki boşlukları değiştirirse M2 scorer bunu hata sayar; eğitim hedefleri kaynağın boşluk düzenini korumalı. GECTurk M2 dosyalarının `S` satırının parquet tokenlarıyla birebir aynı olduğu, M2 dosyaları gelince doğrulanacak.

## Kapsam genişletme kararları

### K25. Sınırlı keşif fazı

Tarih: 2026-11-23 (kullanıcı onayıyla).

- **Karar:** D1 ve D2'den sonra, genellenebilirliği artırmak için en fazla üç teknik denenir. Bütün denemeler doğrulama setlerinde (GECTurk dev, `oscar_dev`) değerlendirilir. Öncelik sırası:
  1. Sentetik gürültü ekleme: Türkçe karaktersiz yazım, klavye hatası, bitişik ve ayrı yazım.
  2. Boyut merdiveni: pilotta seçilen iki küçük model (yaklaşık 0,2B ve 0,75B) D2 ile eğitilir.
  3. Nicemleme kıyası (çıkarımda fp16, 8-bit, 4-bit) ya da GPT GEC verisiyle üçüncü karışım.

  Hangi üçünün deneneceği Kapı B'de kesinleşir. Doğrulama setinde en iyi çıkan yapılandırma bir kez eğitilir ve test setlerinde bir kez ölçülür. Boyut merdiveni ve nicemleme kıyası seçim değil raporlama amaçlıdır; onların test sonuçlarına göre ayar yapılmaz.
- **Neden:** İlk plandaki teknikler dürüst ölçmek ve güvenli eğitmek içindi; modeli daha genellenebilir yapan tek kaldıraç D2'deki organik veriydi. Hedef "hafif ve iyi genellenen düzeltici" olunca bu boşluk doldurulmalı. Sınır (üç teknik, yalnızca doğrulama seti) kapsamın yeniden dağılmasını ve test setine göre ayar yapmayı önler.
- **Bedel:** Tahminen 7–12 GPU saati ve 10–15 saat ek çalışma. Film yorumu ve tweet alanı için doğrulama verisi olmadığından (K16) keşif yalnızca iki doğrulama setiyle yönlendirilir.

### K26. Verimlilik ve genellenebilirlik ölçümleri

Tarih: 2026-11-23 (kullanıcı onayıyla).

- **Karar:** Sonuç tablosuna iki ek ölçüm grubu girer:
  - **Verimlilik:** bizim modeller ve iki temel model için parametre sayısı, eğitilen parametre sayısı, disk boyutu, tepe GPU belleği, eğitim süresi ve saniyede cümle.
  - **Görülmemiş alan:** eğitim kaynaklarının dışındaki bir alandan hatasız metinde yanlış düzeltme oranı ve BOUN de/da testi. Kendi etiketleyeceğimiz 200–300 cümlelik set isteğe bağlıdır ve yedek listede kalır.
- **Neden:** Kullanıcı projeyi danışmana "hafif modeller ve nicemlemeyle eski sonuçlara yaklaşmak" olarak anlattı; bu ancak verimlilik sayılarıyla savunulabilir. "Genellenebilir" iddiası da üç test setinin ötesinde en az bir ölçüm ister.
- **Bedel:** Yaklaşık 6–8 saat ek çalışma. Hatasız metnin kaynağı ve lisansı görev 02'de doğrulanarak seçilecek.

### K27. OSCAR alt kümesine hatasız örnek eklenir

Tarih: 2026-11-23 (kullanıcı kararı). Önerilen oran %50 idi; kullanıcı %25'i seçti.

- **Karar:** OSCAR alt kümesinin bir bölümü hatasız örnektir: başka OSCAR satırlarının düzeltilmiş hali hem kaynak hem hedef olarak kullanılır. Toplam 12.000 olarak kalır. Oran `sampling.oscar_clean_ratio` ile belirlenir; değeri 0,25 (9.000 hatalı + 3.000 hatasız). Hatalı ve hatasız örnekler farklı satırlardan gelir. Ayrıca doğrulama için 250 hatasız örnek ayrılır (`oscar_dev_clean`); 500 cümlelik `oscar_dev` değişmez.
- **Neden:** OSCAR eğitim verisinde kaynağı ve hedefi aynı olan satır %0,08. Hatasız örnek eklenmezse D2'nin hatasız oranı %25'e düşer (D1'de %50) ve model doğru yazılmış organik metne dokunmamayı hiç görmez; hedef ise yanlış düzeltme yapmayan, iyi genellenen bir modeldir. %25 ile organik hata sinyalinin dörtte üçü korunur: OSCAR satırlarında ortalama 3,1 düzeltme var, 9.000 satır 27.835 düzeltme taşır. D2'nin hatasız oranı %25 yerine %37,5 olur.
- **Bedel:** Organik hatalı satır 12.000'den 9.000'e iner; bu, OSCAR testi ve tweetlerde recall'u bir miktar düşürebilir. D1'in hatasız oranı %50, D2'ninki %37,5'tir; iki eğitim arasındaki farkın bir kısmı bu orandan gelebilir ve raporda sınırlılık olarak yazılır. OSCAR hedeflerinin nasıl üretildiği ve ne kadar temiz olduğu doğrulanmadı; hedefte kalmış bir hata "doğru" diye öğretilmiş olur (adım 7'deki elle kontrolde bakılacak). D2 artık "OSCAR'ın hatalı ve hatasız örnekleri eklendi" anlamına gelir. Oran test setlerine bakılarak ayarlanmaz; değiştirilecekse Kapı B'den önce ve doğrulama setleriyle.

