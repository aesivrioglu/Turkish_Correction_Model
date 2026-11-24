# Rapor 01c: Veri raporu (görev 01, adım 4–6)

Durum: OSCAR kısmı 2026-11-23'te, GECTurk kısmı ve bütün çiftlerin sızıntı taraması 2026-11-24'te yazıldı. Eksik kalan tek adım kullanıcının elle kontrolü (adım 7).

## Üretilen dosyalar

Hepsi `data/processed/` altında; depoya girmez. Tek komut: `python scripts/prepare_data.py` (sırayla `convert`, `sample`, `leakage`).

| Dosya | Örnek | Kaynak ve hedef aynı | Rol |
| --- | --- | --- | --- |
| `movie_reviews.jsonl` | 300 | 150 (%50) | Test 1 (kilitli) |
| `oscar_test.jsonl` | 2.408 | 26 (%1,1) | Test 2 (kilitli) |
| `tweets.jsonl` | 2.000 | 89 (%4,5) | Test 3 (kilitli) |
| `gecturk_dev.jsonl` | 20.768 | 10.506 (%50,6) | Doğrulama: model ve eşik seçimi (sentetik) |
| `gecturk_dev_2k.jsonl` | 2.000 | 1.000 (%50) | Pilot ve keşif fazında kullanılan sabit GECTurk doğrulama alt kümesi (katmanlı, tohum 42, kural başına en az 20; `gecturk_dev` içinden) |
| `oscar_dev.jsonl` | 500 | 0 | Doğrulama: model ve eşik seçimi |
| `oscar_dev_clean.jsonl` | 250 | 250 (hepsi) | Doğrulama: organik metinde aşırı düzeltme (K27) |
| `gecturk_train_12k.jsonl` | 12.000 | 6.000 (%50) | D1'in tamamı, D2'nin sentetik yarısı |
| `oscar_train_12k.jsonl` | 12.000 | 3.007 (%25) | D2'nin organik yarısı: 9.000 hatalı + 3.000 hatasız (K27) |
| `gecturk_train.jsonl`, `gecturk_test.jsonl` | 96.919 ve 20.769 | | Alt kümenin kaynağı; GECTurk'ün sentetik testi (kullanılmıyor, yalnızca sızıntı taramasında) |
| `splits/*_ids.txt` | | | Seçilen kimlikler (dört alt küme için) |
| `splits/*_sample_stats.json`, `splits/leakage_report.json` | | | Aşağıdaki sayıların ham hali |
| `pools/oscar_error_pool.jsonl`, `pools/oscar_clean_pool.jsonl` | 13.999 ve 7.438 | | Akıştan seçilen havuzlar; oran değişince veri yeniden okunmaz |

## GECTurk: dönüştürme ve 12 binlik alt küme (2026-11-24)

Kaynak: GECTurk yazarlarının Drive klasöründeki M2 dosyaları (`data/raw/gecturk/`; ayrıntı ve sha256 değerleri `01_veri_kaynaklari.md` içinde).

### Dönüştürme

`trgec.convert.convert_gecturk`: `S` satırı kaynak cümle, hedef dosyasının aynı sıradaki satırı hedef cümle olur. Her düzenlemenin konumu, yanlış ve doğru ifadesi ve kuralı `extra.edits` içinde saklanır; eğitimdeki düzeltme listesi buradan kurulacak. Kural adları M2'deki haliyle tutuluyor (`rule_6`); 10, 11, 19 ve 21 numaralı kuralların adı makalede olmadığı için ada çevrilmedi.

**Doğrulama:** M2 düzenlemeleri kaynak cümleye uygulanınca hedef dosyasındaki cümle elde ediliyor mu?

| Ayrım | Cümle | Birebir aynı | Farklı |
| --- | --- | --- | --- |
| train | 96.919 | 96.919 | 0 |
| dev | 20.769 | 20.769 | 0 |
| test | 20.769 | 20.769 | 0 |

Yani konumları doğru okuyoruz ve iki dosya tutarlı. Bu, hedef dosyası olmayan film yorumlarında hedefi düzenlemelerden üretmenin güvenilir olduğunu da gösteriyor.

| Dosya | Cümle | Hatasız | Düzenleme | Birden çok kural | Kelime: ortalama / %95 / en uzun |
| --- | --- | --- | --- | --- | --- |
| `gecturk_train.jsonl` | 96.919 | 48.748 (%50,3) | 72.997 | 14.104 | 11,2 / 26 / 117 |
| `gecturk_dev.jsonl` | 20.768 | 10.506 (%50,6) | 15.520 | 3.009 | 11,2 / 26 / 153 |
| `gecturk_test.jsonl` | 20.769 | 10.457 (%50,3) | 15.585 | 3.082 | 11,1 / 26 / 81 |
| `movie_reviews.jsonl` | 300 | 150 (%50,0) | 227 | 51 | 15,0 / 34 / 58 |

Gözlemler:

- **Düzenlemesi olup cümlesi değişmeyen satır:** train'de 2, dev'de 1 (düzeltme, yanlış ifadeyle aynı). Train'dekiler eğitim havuzundan atıldı.
- **GECTurk dev'den 1 cümle çıkarıldı:** `Allah rahmet eylesin...` cümlesi tweet setinde de var. Kural gereği çakışma test setinden değil, bizim tarafımızdan çıkarılır; dev 20.768 cümle oldu.
- **Film yorumlarında örtüşen düzenlemeler (3 cümle).** Aynı kelimeye birden çok düzenleme yazılmış: 6. cümlede aynı düzenleme iki kez (`itibariylede` → `itibariyle de`); 75. cümlede iki adım (`izlenilmeseda` → `izlenilmesede`, sonra → `izlenilmese de`); 125. cümlede iç içe iki düzenleme (`halbu` → `Halbu` ve `halbu ki` → `Halbuki`). Tek bir hedef cümle için kural: örtüşenlerden en geniş olan, eşitse dosyada sonra yazılan uygulanır. Üçünde de sonuç son ve tam düzeltme (`itibariyle de`, `izlenilmese de`, `Halbuki`). Bu cümleler `extra.overlap_resolved` ile işaretli; bütün düzenlemeler `extra.edits` içinde duruyor. Bu yalnızca ERRANT-TR için üretilen hedef cümleyi etkiler; M2 scorer özgün M2 dosyasını kullanır.

### 12 binlik katmanlı alt küme (`gecturk_train_12k.jsonl`)

D1'in tamamı ve D2'nin sentetik yarısı. `trgec.sampling.stratified_sample`, tohum 42, `clean_ratio` 0,5, `min_per_rule` 50.

| Adım | Sayı |
| --- | --- |
| GECTurk train | 96.919 |
| Düzenlemesi olup cümlesi değişmeyen (atıldı) | 2 |
| Yineleme: aynı normalleştirilmiş kaynak cümle (atıldı) | 293 |
| GECTurk dev ile çakışan (atıldı) | 100 |
| GECTurk test ile çakışan (atıldı) | 84 |
| Film yorumları, OSCAR testi, tweetler ile çakışan | 0 / 0 / 0 |
| Havuz | 96.440 |
| Seçilen | 12.000 (6.000 hatalı + 6.000 hatasız) |

Alt kümede 9.513 düzenleme var; 1.994 cümle birden çok kural içeriyor. Kaynak kelime sayısı: ortalama 11,4, ortanca 9, %95'lik dilim 27, en uzun 103 (70 kelimeyi aşan 5 cümle; GECTurk'e kelime süzgeci uygulanmıyor, 512 token tavanını aşan örnek görev 03'te elenecek).

Kural sıklıkları (kuralın geçtiği cümle sayısı):

| Kural | Train | Alt küme | | Kural | Train | Alt küme |
| --- | --- | --- | --- | --- | --- | --- |
| `rule_1` | 8.592 | 1.003 | | `rule_14` | 605 | 101 |
| `rule_2` | 72 | 50 | | `rule_15` | 228 | 67 |
| `rule_3` | 66 | 51 | | `rule_16` | 286 | 78 |
| `rule_4` | 330 | 74 | | `rule_17` | 322 | 77 |
| `rule_5` | 7.261 | 873 | | `rule_18` | 396 | 93 |
| `rule_6` | 21.875 | 2.681 | | `rule_19` | 3 | **3** |
| `rule_7` | 1.251 | 182 | | `rule_20` | 5.110 | 622 |
| `rule_8` | 976 | 167 | | `rule_21` | 38 | **38** |
| `rule_9` | 197 | 70 | | `rule_22` | 2.525 | 299 |
| `rule_10` | 266 | 79 | | `rule_23` | 1.828 | 247 |
| `rule_11` | 3.681 | 431 | | `rule_24` | 583 | 136 |
| `rule_12` | 8.033 | 940 | | `rule_25` | 193 | 68 |
| `rule_13` | 1.086 | 156 | | | | |

- **Her kuraldan en az 50 cümle var; iki istisna** train'de zaten 50'den az olan `rule_19` (3) ve `rule_21` (38): ikisinin de hepsi alındı. `rule_19` üç örnekle fiilen öğrenilemez; kural bazlı analizde sınırlılık olarak yazılacak.
- **Katmanlama nadir kuralları koruyor.** Düz rastgele seçimde `rule_2` yaklaşık 9, `rule_3` yaklaşık 8 cümleyle kalırdı (12.000 / 96.919 oranıyla); şimdi 50 ve 51.
- **Alt küme yeniden üretilebilir:** adım iki kez çalıştırıldı, dosya bayt bayt aynı. Kimlikler `splits/gecturk_train_12k_ids.txt`, sayılar `splits/gecturk_sample_stats.json` içinde.

## OSCAR eğitim verisi nasıl seçildi?

Kaynak: `asimokby/Turkish-OSCAR-GEC`, `train.csv`, sürüm `0f90263` (2024-05-08). Dosya diske kaydedilmedi; akış halinde bir kez baştan sona okundu.

| Adım | Sayı |
| --- | --- |
| `train.csv` içindeki satır | 1.737.696 |
| Boş ya da çok satırlı alan | 0 |
| Kaynak cümlesi 70 kelimeden uzun (elendi, K11 notu) | 79.929 (%4,6) |
| Uygun satır | 1.657.767 |
| Uygun satırlar içinde kaynak ve hedef aynı | 1.257 (%0,08) |
| **Hatalı havuz:** rezervuar örneklemeyle seçilen (tohum 42; 500 + 12.000 + 1.500 yedek) | 14.000 |
| Yineleme (aynı normalleştirilmiş kaynak) | 1 |
| Test setleriyle çakışan | 0 |
| Hatalı havuzda kalan | 13.999 |
| **Hatasız havuz:** aynı akıştan bağımsız ikinci örnek (tohum 43) | 7.500 |
| Hatalı havuzda da bulunan satır (atıldı) | 60 |
| Yineleme | 1 |
| Hatalı havuzdaki bir cümleyle aynı metin (atıldı) | 1 |
| Test setleriyle çakışan | 0 |
| Hatasız havuzda kalan | 7.438 |
| Kullanılan | `oscar_dev` 500 hatalı; `oscar_dev_clean` 250 hatasız; eğitim 9.000 hatalı + 3.000 hatasız |

Notlar:

- **Satır sayısı.** Belgelerde geçen "2,3 milyon çift" bütün veri setinin boyutudur. `train.csv` 1,74 milyon satır; kalan kısım `val.csv` içinde (251 MB, sayılmadı, kullanılmıyor).
- **Örnek dosyanın tamamına yayılmış.** Seçilen satır numaralarının çeyrekleri 436.748, 874.119 ve 1.298.667; eşit dağılımda beklenen 434 bin, 869 bin ve 1,30 milyon.
- **Kimlik satır numarasını taşır.** `oscar-train-513614`, `train.csv` içindeki 513.614. satırdır. Kimlik listesi ve sabit sürümle alt küme yeniden üretilebilir.
- **Hatasız örnek nasıl üretiliyor?** Hatasız havuzdaki satırın yalnızca düzeltilmiş hali alınır ve hem kaynak hem hedef olarak yazılır. Kimliğin sonunda `-c` bulunur (`oscar-train-633428-c`) ve `extra.clean_from_target` işaretlidir.
- **Üç dosya ayrık:** eğitim, `oscar_dev` ve `oscar_dev_clean` arasında ortak satır ve ortak normalleştirilmiş cümle yok. Eğitimin hatalı ve hatasız yarıları da farklı satırlardan geliyor.
- **Oran değişse de seçim iç içe kalır.** Eğitimdeki 9.000 hatalı satır, hatalı havuzun `oscar_dev`'den sonraki ilk 9.000 satırıdır; oran %50 olsaydı bunların ilk 6.000'i kullanılırdı. `oscar_dev` oran değişiminden etkilenmez.

## Sızıntı taraması (bütün çiftler, 2026-11-24)

`python scripts/prepare_data.py --step leakage` üretilmiş dosyaları bağımsız olarak yeniden tarar; çakışma bulursa hata verir. Karşılaştırma: kaynak ve hedef cümleler, `normalize_for_match` (Türkçe küçük harf, noktalama silme) sonrası tam eşleşme.

| Bizim taraf | Karşı taraf | Çakışma |
| --- | --- | --- |
| `gecturk_train_12k` (12.000) | GECTurk dev, GECTurk test, film yorumları, OSCAR testi, tweetler | 0 / 0 / 0 / 0 / 0 |
| `oscar_train_12k` (12.000) | `oscar_dev`, `oscar_dev_clean`, film yorumları, OSCAR testi, tweetler, GECTurk dev, GECTurk test | hepsi 0 |
| `oscar_dev` (500) | film yorumları, OSCAR testi, tweetler | 0 / 0 / 0 |
| `oscar_dev_clean` (250) | film yorumları, OSCAR testi, tweetler | 0 / 0 / 0 |
| `gecturk_dev` (20.768) | film yorumları, OSCAR testi, tweetler | 0 / 0 / 0 (tweetlerle çakışan 1 cümle çıkarıldıktan sonra) |

Film yorumları artık kaynak ve hedef cümleleriyle taranıyor (önceki taramada yalnızca kaynak vardı); OSCAR alt kümeleri bu yüzden değişmedi (dosya bayt bayt aynı).

Sonuç dosyası: `splits/leakage_report.json`.

## Sızıntı taraması (OSCAR havuzları, 2026-11-23)

İki havuzdaki satırların kaynak ve hedef cümleleri, test setlerinin kaynak ve hedef cümleleriyle `normalize_for_match` (Türkçe küçük harf, noktalama silme) sonrası tam eşleşme ile karşılaştırıldı.

| Eğitim tarafı | Test tarafı | Çakışma |
| --- | --- | --- |
| OSCAR hatalı havuz (13.999) ve hatasız havuz (7.438) | OSCAR GEC test (kaynak ve hedef) | 0 |
| İki havuz | Tweetler (kaynak ve hedef) | 0 |
| İki havuz | Film yorumları (yalnızca kaynak; hedefler M2'de) | 0 |

Sınırlılıklar:

- **Eşleşme satırın tamamı üzerinden.** OSCAR satırları çoğu zaman birkaç cümle içerir; bir test cümlesinin daha uzun bir eğitim satırının içinde geçmesi bu taramayla yakalanmaz.
- Film yorumlarının hedef cümleleri ve GECTurk tarafı 2026-11-24'te tarandı (yukarıdaki tablo).

## Seçilen alt kümenin özellikleri

| | `oscar_dev` | `oscar_dev_clean` | `oscar_train_12k` |
| --- | --- | --- | --- |
| Örnek | 500 | 250 | 12.000 |
| Kaynak ve hedef aynı | 0 | 250 | 3.007 (3.000 eklenen + 7 doğal) |
| Kaynak kelime sayısı: ortalama / ortanca / %95 / en uzun | 21,2 / 18 / 51 / 70 | 21,1 / 18 / 47 / 67 | 21,3 / 18 / 50 / 73 |
| Hedefi 70 kelimeden uzun | 1 | 0 | 16 |

Süzgeç kaynak cümleye uygulanıyor; hatasız örnek düzeltilmiş cümleden üretildiği için birkaçı 70 kelimeyi biraz aşıyor (en uzun 73).

Hata yoğunluğu (K27'den önceki 12.000 hatalı satırda, kelime düzeyinde fark): satır başına ortalama 3,1, ortanca 2 düzeltme; toplam 37.037. Eğitime giren 9.000 hatalı satırda 27.835 düzeltme var.

Tam eğitim örneğinin token uzunluğu (talimat + cümle + hedef, sohbet şablonuyla; üç dosya birlikte 12.750 örnek):

| Tokenizer | Ortalama | %95 | %99 | En uzun | 384'ü aşan | 512'yi aşan |
| --- | --- | --- | --- | --- | --- | --- |
| Kumru-2B | 115 | 215 | 277 | 507 | 7 | 0 |
| Qwen3 (üç model) | 180 | 322 | 403 | 603 | 207 | 3 |
| EuroLLM-1.7B | 169 | 297 | 372 | 650 | 82 | 2 |
| Kanarya-2b | 114 | 209 | 263 | 519 | 3 | 1 |
| Cosmos GPT-2 large | 111 | 202 | 255 | 484 | 2 | 0 |
| ufakzeka-1 | 115 | 215 | 271 | 502 | 7 | 0 |

512 tavanı 12.750 örneğin en fazla 3'ünü dışarıda bırakıyor. Süzgeçten sonra ortalama örnek de kısaldı (Kumru'da 140'tan 115'e, Qwen3'te 211'den 180'e); `raporlar/03_model_adaylari.md` içindeki D2 süre tahminleri bu yüzden biraz yüksek kalıyor.

## Hatasız cümle oranı ve K27

OSCAR eğitim verisinde kaynağı ve hedefi aynı olan satır %0,08 (1.657.767 uygun satırda 1.257). Hatasız örnek eklenmeden seçilen 12.000 satırda yalnızca 9 hatasız cümle vardı; `oscar_dev` içinde hiç yok.

Kullanıcı 2026-11-23'te hatasız örnek eklenmesine karar verdi ve oranı %25 olarak seçti (K27). Önerilen oran %50 idi.

| | Hatasız örnek eklenmeden | %25 (seçilen) | %50 (seçilmedi) |
| --- | --- | --- | --- |
| OSCAR yarısı | 11.991 hatalı + 9 hatasız | 8.993 hatalı + 3.007 hatasız | 5.997 hatalı + 6.003 hatasız |
| D2'nin tamamında hatasız oranı (GECTurk yarısı %50; 2026-11-24'te doğrulandı) | %25 | %37,5 | %50 |
| D1'de hatasız oranı | %50 | %50 | %50 |
| Organik düzeltme sayısı | 37.037 | 27.835 | 18.618 |

Riskler:

- **Organik hatalı satır dörtte bir azaldı.** OSCAR testi ve tweetlerde recall bir miktar düşebilir.
- **D1 ve D2'nin hatasız oranı farklı** (%50 ve %37,5). İki eğitim arasındaki farkın bir kısmı bu orandan gelebilir; raporda sınırlılık olarak yazılacak.
- **OSCAR hedeflerinin temizliği doğrulanmadı.** Hedefte kalmış bir hata, hatasız örnekte "doğru" diye öğretilir. Adım 7'deki elle kontrolde hatasız örneklere ayrıca bakılmalı. **2026-11-24 gözlemi:** `show_examples.py` ile bakılan üç hatalı OSCAR örneğinin üçünde de hedefte düzeltilmemiş hata kaldı (`ayni`, `yasasak`, `oglak.su donem`, `yapiyordum.ama tukendim`). Yani OSCAR hedefleri "gümüş" kalitede: kısmen düzeltilmiş. Üç örnek bir oran vermez; elle kontrolde 20 hatasız örnekte kaç tanesinde hata kaldığı sayılmalı ve sonuç K27 için değerlendirilmeli.
- **Oran bir ayar.** Test setlerine bakılarak seçilmez; değişecekse Kapı B'den önce ve doğrulama setleriyle.

## Lisans özeti

Ayrıntı `01_veri_kaynaklari.md` içinde. Özet: GECTurk'ün HF veri seti Apache-2.0 (GitHub deposunda lisans yok; M2 dosyaları yazarların Drive klasöründen); OSCAR GEC eğitim ve test verisi MIT; #Turki$hTweets lisanssız (yalnızca değerlendirmede, yeniden dağıtmadan kullanılıyor). Hiçbir veri depoya girmiyor.

## Bekleyenler

- **Adım 7, elle kontrol (kullanıcı):** her setten 20 örnek. Komut: `python scripts/show_examples.py --set <ad>`; hatasız OSCAR örnekleri için `--set oscar_train_12k --only clean`.
- Tam eğitim örneklerinin token uzunluğu GECTurk'ün gerçek hedefleriyle yeniden ölçülecek (görev 03, adım 3).
