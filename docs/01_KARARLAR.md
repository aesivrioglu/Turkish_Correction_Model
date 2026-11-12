# Kararlar ve nedenleri

Bu belge projede alınan kararları, her birinin nedenini ve neye mal olduğunu kaydeder. Bir karar değişirse eski madde silinmez; altına tarihli bir not düşülür.

Her madde üç bölümden oluşur: **Karar**, **Neden**, **Bedel** (bu kararla vazgeçtiğimiz şey).

## Kapsam kararları

### K1. Kapsam çekirdeğe indirildi

- **Karar:** İlk plandaki dört model ailesi, beş deney grubu ve açıklama damıtma yerine tek model, iki eğitim ve üç test seti.
- **Neden:** İlk plan yaklaşık 26 hafta ve 85 GPU saati tutuyordu. Aynı dönemde iki proje daha yürüyor. Her şeye biraz dokunup hiçbirini bitirememek en büyük riskti.
- **Bedel:** Bulgunun başka modellerde de geçerli olup olmadığını göstermiyoruz. Çıkarılan işler `02_IS_PLANI.md` içindeki yedek listede duruyor.

### K2. Genişlik kesildi, değerlendirme derinliği korundu

- **Karar:** Model ve varyant sayısı azaltıldı; üç gerçek test seti, kural bazında analiz ve güven aralıkları kaldı.
- **Neden:** Projeyi "küçük bir modeli QLoRA ile eğittim, demo yaptım" seviyesine indirmek Text Mining projesinden farkını yok ederdi. Ayırt edici olan titiz değerlendirme.
- **Bedel:** Değerlendirme hattı projenin teknik olarak en zor kısmı olarak kalıyor.

### K3. Araştırma soruları ikiye indirildi

- **Karar:** Yalnızca AS1 (performans) ve AS2 (veri karışımı). Açıklama damıtma sorusu çıkarıldı, aşırı düzeltme bir metrik oldu.
- **Neden:** Bir soruyu temiz cevaplamak, dört soruyu yarım cevaplamaktan daha güçlü bir capstone çıkarır.
- **Bedel:** "Açıklama üretmek düzeltme başarısını etkiler mi?" sorusu cevapsız kalıyor.

### K4. Üç gerçek test seti kullanılacak

- **Karar:** GECTurk film yorumları (300 cümle), OSCAR GEC test (2.400 cümle), #Turki$hTweets (2.000 tweet).
- **Neden:** GECTurk'ün kendi sentetik testi doymuş durumda (etiketleyici F0.5 = 0,98), modelleri ayırt edemiyor. Tek gerçek set olan film yorumları 300 cümle ve istatistiksel gücü düşük.
- **Bedel:** Üç farklı biçim ve etiket şemasını tek hatta toplamak gerekiyor.

## Model ve eğitim kararları

### K5. Tek model, iki eğitim (D1 ve D2)

- **Karar:** D1 yalnızca GECTurk alt kümesiyle, D2 aynı GECTurk alt kümesi artı OSCAR GEC alt kümesiyle eğitilir.
- **Neden:** AS2'nin cevabı bu iki eğitimin farkından çıkar. GECTurk kısmı iki eğitimde birebir aynı olduğu için fark OSCAR verisine bağlanabilir.
- **Bedel:** D2 iki kat veri görür; "organik veri mi yardım etti, yoksa daha fazla veri mi?" sorusu tam ayrışmaz. Raporda sınırlılık olarak yazılır. Eşit boyutlu kontrol deneyi yedek listede.

### K6. Eğitim küçültüldü

- **Karar:** 1 ile 2 milyar parametre arası model, GECTurk'ten 12 bin ve OSCAR'dan 12 bin cümle, 1 epok.
- **Neden:** Uzun eğitimler riskli: oturum 12 saatte kesilir, bir hata saatlerce emeği siler. GECTurk makalesinde eğitim verisinin dörtte biriyle bile yüksek skor alınıyor, yani tüm veri şart değil.
- **Bedel:** Daha büyük model ya da daha çok veriyle alınabilecek birkaç puan. Sayılar pilotta ölçülen hıza göre güncellenir.

### K7. Model seçimi mini eğitimle yapılır, sıfır-atışla değil

- **Karar:** İki aday model 2 bin örnekle kısa birer eğitimden geçirilir ve doğrulama setinde kıyaslanır.
- **Neden:** Küçük modellerin eğitimsiz başarısı eğitim sonrasını iyi öngörmez. Sıfır-atışta kötü görünen model ince ayardan sonra daha iyi olabilir.
- **Bedel:** Yaklaşık 1 GPU saati ek maliyet.

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
