# Rapor 02: Skorlayıcı (görev 02, adım 1 ve 6)

Tarih: 2026-11-23. Durum: ERRANT-TR kuruldu ve doğrulandı; `scripts/evaluate.py` OSCAR testi, tweetler ve iki OSCAR doğrulama diliminde çalışıyor. **Kapı A geçildi:** mT5 tweetlerde F0.5 78,5 (yayımlanan 79,0), OSCAR testinde 55,1 (yayımlanan 55,4), film yorumlarında 28,8 (yayımlanan 28,1) verdi. Ana karşılaştırma modu `cs` olarak sabitlendi (2026-11-24); `cse` yanında raporlanıyor.

## Kısa sonuç

- **ERRANT-TR çalışıyor ve yayımlanmış referansı birebir üretiyor.** OSCAR GEC testinin resmi M2 dosyası, `inp.txt` ve `ref.txt` dosyalarından yeniden üretildi: 2.408 bloğun 2.408'i birebir aynı.
- **Referans dosyasının nasıl üretildiği artık belli:** ERRANT-TR `process.py`, ardından `map_labels.py`. Bizim hipotezlerimiz de aynı iki adımdan geçecek.
- **Skorlama cümle başına (tp, fp, fn) veriyor;** bootstrap güven aralığına doğrudan bağlanabilir (K18).

## Kurulum

| Parça | Sürüm | Not |
| --- | --- | --- |
| ERRANT-TR | `github.com/harunuz/erranttr` @ `56cf0a3` (2023-08-17), MIT | `external/erranttr`; depoya girmez |
| `tr_TR.aff` | sürüm v0.1.0; 2.283.891 bayt; SHA-256 `a04a227b2ee4…` | `erranttr/tr/resources/` altında |
| `TS_Corpus_Turkish_Word_List.txt` | sürüm v0.1.0 (zip içinden); 49.523.542 bayt; SHA-256 `efd432b72be0…` | aynı klasörde |
| `zemberek-python` | 0.2.3 | `antlr4-python3-runtime` 4.8 ve `numpy` ile gelir |
| `rapidfuzz` | 3.14.6 | |
| `setuptools` | 80.10.2 (`<81`) | aşağıya bak |
| Python | 3.13.4 | |

Kurulum adımları `docs/KURULUM.md` içinde.

İki uyumluluk sorunu çıktı ve çözüldü:

1. **`pkg_resources` yok.** `zemberek-python` bu modülü kullanıyor; modül `setuptools` ile gelir ama sanal ortamda `setuptools` kurulu değildi ve 81 sonrası sürümlerde modül yok. Çözüm: `setuptools<81` bağımlılıklara eklendi. Bu, onaylanan iki bağımlılığın çalışması için gereken üçüncü bir pakettir.
2. **`typing.io` yok.** `antlr4-python3-runtime` 4.8 bu modülü ister; Python 3.13'te kaldırıldı. `zemberek-python` 4.8'e sabitlendiği için sürüm yükseltilemez. Çözüm: `trgec.errant_tr._patch_typing_io` aynı adları taşıyan boş bir modül tanıtıyor. Kurulu Python 3.11 ve 3.12 ile ayrı bir ortam kurmak da mümkündü; tek ortam daha basit olduğu için yama seçildi.

## Yazarların yolu

ERRANT-TR deposundaki `README.md` ve `pipe.sh` üç adım tarif ediyor:

```
python process.py predictions            # paralel cümleler -> M2
python map_labels.py predictions         # ERRANT-TR türleri -> ERRANT türleri
python commands/compare_m2.py -hyp ... -ref ... -b 0.5 -cse -cat 2
```

- `process.py` sabit bir dosya (`tr/resources/sentences.txt`, sekmeyle ayrılmış kaynak ve düzeltme) okur ve Zemberek ile tokenize eder. Kaynak ve düzeltme aynıysa `noop` satırı yazar.
- `commands/parallel_to_m2.py` İngilizce ERRANT'tan kopyalanmış ve Türkçeye uyarlanmamış; kullanılmıyor.
- `-cse`, bir düzeltmenin doğru sayılması için konum ve metnin yanında **hata türünün de** eşleşmesini ister. `-cs` (ERRANT'ın varsayılanı) türe bakmaz.

**Bilinmeyen:** Ersoy ve Yıldız hangi modu kullandı? Makale yalnızca "ERRANT's evaluation scripts" diyor; yazarların deposunda komut yok. ERRANT-TR'nin kendi örneği `-cse` kullanıyor. Kapı A bunu ayırt etmedi: iki mod da yayımlanmış skorları üretiyor (aşağıda).

**Karar (2026-11-24, K15 notu): ana skor `cs`.** Gerekçe:

1. İki mod pratikte aynı sayıyı veriyor: tweetlerde 6.277 düzenlemede 1, OSCAR testinde 22.366 düzenlemede 2 düzenleme farklı sayılıyor (F0.5 farkı 0,02 puan). Seçim sonuçları değiştirmez.
2. `cs`, ERRANT'ın varsayılanı ve GEC yazınındaki standart ölçüdür (konum ve metin eşleşmesi). Jüriye ve okura açıklaması en kolay olan budur.
3. `cse` ana skoru hata türü etiketine bağlar. Etiketler `map_labels` adımından geçiyor ve bu adımın bilinen bir yan etkisi var (aşağıda). Düzeltmenin doğruluğunu ölçen sayının, etiketleme ayrıntısına bağlı olmaması daha sağlam.
4. Projenin amacı cümleyi doğru düzeltmek; hata türünü doğru adlandırmak ayrı bir soru ve hata türü analizinde (iş paketi 5) ayrıca ele alınıyor.

`evaluate.py` `cse` sonucunu da hesaplayıp aynı dosyaya yazmayı sürdürüyor; ana skor `primary_mode` alanında belirtiliyor.

## Doğrulama: resmi M2 dosyasını yeniden üretmek

Komut: `python scripts/check_errant_tr.py` (yaklaşık 14 dakika; `--limit 60` ile 15 saniye).

Girdi olarak yalnızca altın kaynak (`inp.txt`) ve altın düzeltme (`ref.txt`) kullanıldı; model çıktısı yok.

| Ölçüm | Sonuç |
| --- | --- |
| Satır | 2.408 |
| Hata veren satır | 0 |
| Birebir aynı blok (`S` satırı ve bütün `A` satırları) | 2.408 / 2.408 |
| Yeniden üretilen M2, resmi M2'ye karşı (`cs`) | TP 22.366, FP 0, FN 0; F0.5 = 1,0000 |
| Yeniden üretilen M2, resmi M2'ye karşı (`cse`) | TP 22.366, FP 0, FN 0; F0.5 = 1,0000 |
| Süre | 817 saniye (satır başına yaklaşık 0,34 saniye; iki cümle çözümleniyor) |

Bu üç şeyi birden gösteriyor: kurulum doğru, sürümler yazarların kullandığıyla aynı sonucu veriyor ve referans dosyası tam olarak bu yolla üretilmiş.

## `map_labels.py` yan etkisi

`map_labels.py` türleri alan alan değil, bütün dosyada düz metin değiştirmeyle eşliyor. Bu yüzden cümlenin ve düzeltmenin içindeki büyük harfli parçalar da değişiyor: `DAMACANA` → `SPELLMACANA`, `TENDA` → `TENSPELL`. Resmi OSCAR M2 dosyasında 13 `S` satırında bu iz var.

Sonucu: hipotez M2'si de aynı işlemden geçmezse, içinde büyük harfli `DA` ya da `ACCENT` geçen düzeltmeler referansla eşleşmez ve doğru düzeltme yanlış sayılır. `trgec.errant_tr.map_labels` bu davranışı bilerek aynen koruyor (K15: yayımlanmış skorun aracı kullanılır). Raporda sınırlılık olarak belirtilmeli.

## Sarmalayıcı: `src/trgec/errant_tr.py`

| İşlev | Karşılığı | Ne yapar |
| --- | --- | --- |
| `load_annotator` | `erranttr.load("tr")` | Kurulumu denetler, yamayı uygular, Zemberek'i yükler (yaklaşık 7 saniye) |
| `parallel_to_m2` | `process.py` | Kaynak ve düzeltme listelerinden M2 metni üretir; dosya adı dayatmaz |
| `map_labels` | `map_labels.py` | Tür eşlemesi (yan etkisiyle birlikte) |
| `sentence_counts` | `commands/compare_m2.py` | ERRANT-TR'nin kendi sayım işlevlerini çağırır; toplam yerine cümle başına (tp, fp, fn) döndürür |
| `total_counts` | | Cümle bazlı sayıları toplar |

`process.py`'den tek fark: boş kaynak satırı atlanmaz, hata verilir; atlanan satır referansla hizayı kaydırırdı.

Testler (`tests/test_errant_tr.py`, 9 test): altın düzeltme hipotez olunca F0.5 = 1; kaynak cümle hipotez olunca recall = 0; gereksiz düzeltme yanlış pozitif sayılıyor; tür farkı yalnızca `cse` modunda sonucu değiştiriyor; `map_labels` özgün betikle aynı sonucu veriyor. Kurulum yoksa testler atlanır.

## `scripts/evaluate.py`

```
python scripts/evaluate.py --predictions outputs/predictions/<calisma>/tweets.jsonl --testset tweets
python scripts/evaluate.py --baseline source --testset oscar_test      # akıl sağlığı koşusu
```

Adımlar:

1. Tahmin dosyası (`{"id", "hypothesis"}`) test kayıtlarıyla kimlik üzerinden hizalanır. Eksik, fazla ya da yinelenen kimlik hata verir; sessizce atlanan cümle skoru olduğundan iyi gösterirdi.
2. Sete özgü dönüşüm uygulanır (yalnızca tweetlerde; aşağıya bak).
3. Referans M2 alınır: OSCAR testinde resmi dosya; resmi dosyası olmayan setlerde kaynak ve altın cümlelerden üretilir ve `outputs/eval/_ref/` altında saklanır.
4. Hipotez M2'si üretilir (`parallel_to_m2` + `map_labels`). ERRANT-TR bir cümlede hata verirse o cümle "değişiklik yok" sayılır ve sayısı raporlanır.
5. `cs` ve `cse` modlarının ikisi de hesaplanır: P, R, F0.5 ve 1.000 tekrarlı bootstrap ile %95 güven aralığı.
6. Hatasız cümlelerde (kaynak = altın) aşırı düzeltme oranı ve, tahmin dosyasında `well_formed` alanı varsa, bozuk çıktı oranı eklenir.
7. Sonuç `outputs/eval/<calisma>/<set>.json` dosyasına, hipotez M2'si yanına yazılır. JSON cümle başına (tp, fp, fn) sayılarını da içerir; iki modelin eşleştirilmiş karşılaştırması bunlardan yapılacak.

Hazır setler: `oscar_test`, `tweets`, `movie_reviews`, `oscar_dev`, `oscar_dev_clean` (hepsi ERRANT-TR). `gecturk_dev` ve film yorumlarının M2 scorer tarafı M2 scorer'ın bağlanmasını bekliyor.

## Tweet dönüşümleri

Tweet setinde altın cümleler küçük harfle başlıyor ve cümle noktalaması içermiyor; mT5 ise büyük harfle başlayan, noktalamalı cümle üretiyor. Ersoy ve Yıldız bu yüzden altın cümlenin ilk harfini büyütüyor ve model çıktısındaki noktalamayı siliyor. Dönüşümler `trgec.evaluation.tweet_protocol` içinde; `data/processed/tweets.jsonl` değişmiyor (K24).

Setin biçimi (yalnızca biçime bakıldı):

| Gözlem | Sayı |
| --- | --- |
| Altın cümlenin ilk karakteri: küçük harf / büyük harf / `#` / rakam | 1.686 / 229 / 46 / 39 |
| Küçük `i` ya da `ı` ile başlayan altın cümle | 55 |
| İlk kelime dışında büyük harfle başlayan kelime içeren altın cümle | 407 |
| Altın cümlelerdeki noktalama | yalnızca `'` (367), `#` (76), `’` (2) |

Varsayılan ayar ve Kapı A'da denenecek seçenekler (`configs/base.yaml`, `eval.tweets`):

| Ayar | Varsayılan | Diğer seçenekler | Etkilediği cümle |
| --- | --- | --- | --- |
| `capitalize` | `turkish` (`i` → `İ`) | `ascii` (`i` → `I`), `python_capitalize` (ilk harf büyür, gerisi küçülür), `none` | `ascii`: 55; `python_capitalize`: 407'ye kadar |
| `keep_punct` | `'’#` (altın cümlelerde bulunanlar) | `'` , boş | kesme işareti ve `#` içeren çıktılar |

Varsayılan dönüşümle referansta 6.277 düzenleme var; 1.717 cümlede ilk kelimede düzenleme bulunuyor. Yani ilk harfin nasıl büyütüldüğü skoru doğrudan etkiliyor.

## Akıl sağlığı koşuları (model yok)

| Set | Hipotez | TP | FP | FN | P | R | F0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tweetler | altın cümle (dönüştürülmüş) | 6.277 | 0 | 0 | 1,00 | 1,00 | 1,00 |
| Tweetler | kaynak cümle | 0 | 0 | 6.277 | 1,00 | 0,00 | 0,00 |
| OSCAR test | altın cümle (`check_errant_tr.py`) | 22.366 | 0 | 0 | 1,00 | 1,00 | 1,00 |
| OSCAR test | kaynak cümle | 0 | 0 | 22.366 | 1,00 | 0,00 | 0,00 |
| `oscar_dev_clean` | kaynak cümle | 0 | 0 | 0 | 1,00 | 1,00 | 1,00 |

`cs` ve `cse` modları bu koşularda aynı sonucu veriyor. Hiçbir cümlede skorlayıcı hata vermedi. Süreler: tweetlerde bir koşu yaklaşık 3,5 dakika, OSCAR testinde kaynak koşusu yaklaşık 5 dakika.

## Kapı A hedefleri ve mT5'in ayarları

Yazarların deposundaki README (`github.com/asimokby/Turkish-GEC`) F0.5 yanında precision ve recall da veriyor. Bunlar skoru tutturamazsak sorunun hangi yanda olduğunu gösterir:

| Model: OSCAR GEC (mT5) | P | R | F0.5 |
| --- | --- | --- | --- |
| Tweetler (Kapı A hedefi) | 85,1 | 61,3 | 79,0 |
| OSCAR GEC test | 68,7 | 31,2 | 55,4 |
| Film yorumları | 30,0 | 22,5 | 28,1 |

Model: `asimokby/Turkish-OSCAR-GEC-v0`, sürüm `3467f87` (2024-04-28). Yapılandırma dosyalarından doğrulananlar:

- `MT5ForConditionalGeneration`, mT5-base boyutunda (12 + 12 katman, `d_model` 768), fp32, `transformers` 4.35.2 ile kaydedilmiş; eğitim klasörü adı `mt5-Base-finetuned_v9/checkpoint-133672`.
- **`generation_config.json` içinde üretim ayarı yok** (`num_beams`, `max_length`, `max_new_tokens` tanımsız). `transformers`'ın varsayılanı açgözlü kod çözme ve 20 tokenlık çıktıdır; 20 token OSCAR satırlarını keser. Yazarlar bunu mutlaka değiştirmiştir ama değerleri bilinmiyor.
- Model kartı, görev öneki ve örnek kullanım kodu yok.

Kapı A'da denenecek değişkenler, sırayla: çıktı uzunluğu sınırı, ışın sayısı (1 ve 4–5), giriş öneki (yok), tweet dönüşüm ayarları, karşılaştırma modu (`cs` ya da `cse`).

## Kapı A sonucu (2026-11-23)

mT5 yerelde, CPU ile çalıştırıldı (`scripts/predict_seq2seq.py`); Kaggle gerekmedi. Ayarlar: açgözlü kod çözme (`num_beams` 1), `max_new_tokens` 256, giriş öneki yok, girdi kesme yok. Tweet dönüşümleri varsayılan haliyle (`capitalize: turkish`, `keep_punct: "'’#"`).

| Tweetler (2.000 cümle) | TP | FP | FN | P | R | F0.5 | %95 güven aralığı |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Yayımlanan (Ersoy ve Yıldız) | | | | 85,1 | 61,3 | 79,0 | |
| Bizim hat, `cs` | 3.836 | 701 | 2.441 | 84,6 | 61,1 | **78,5** | 77,4–79,6 |
| Bizim hat, `cse` | 3.835 | 702 | 2.442 | 84,5 | 61,1 | 78,5 | 77,4–79,6 |

**Fark 0,5 puan; Kapı A'nın ±2 ölçütü sağlandı.** Precision 0,5, recall 0,2 puan düşük; yayımlanan skor bizim güven aralığımızın içinde. İlk denenen ayar tuttuğu için başka ayar (ışın sayısı, büyütme biçimi, korunan noktalama) denenmedi; test setinde gereksiz deneme yapılmadı.

Bu koşudan öğrenilenler:

- **`cs` ve `cse` neredeyse aynı:** 6.277 düzenlemede yalnızca 1 düzenleme farklı sayılıyor. Hangisinin kullanıldığı skoru ayırt etmiyor; ana skor `cs` olarak sabitlendi (yukarıda, karar).
- **Çözme biçimi tweetlerde etkisiz:** tokenizer'ın "temizleme" adımı (noktalamadan önceki boşluğu siler) 2.000 çıktının hiçbirini değiştirmedi. `oscar_dev` diliminde 500 çıktının 23'ünü değiştiriyor (`%40'ının ,` → `%40'ının,`). Tahmin dosyasında iki hal de saklanıyor (`hypothesis`, `raw_output`); `evaluate.py --hypothesis-field raw_output` ile çıkarım yinelenmeden skorlanır.
- **Çıktı uzunluğu sınırı tweetlerde etkisiz:** hiçbir çıktı sınıra takılmadı; en uzun çıktı 48 token.
- **mT5 uzun cümleleri çoğunlukla 48 tokenda kendisi bitiriyor.** `oscar_dev` diliminde girdisi 48 tokenı aşan 186 cümlenin 137'sinde çıktı tam 48 tokenda bitiş tokenıyla (EOS) sonlandı; kalanında model devam etti (en uzun çıktı 115 token). Bu, modelin 48 tokenda kesilmiş hedeflerle eğitildiğini düşündürüyor (kanıt değil, gözlem). Etkisi setlere göre çok farklı:

| Set | Kaynak cümle, mT5 tokenı (ortalama / %95 / en uzun) | 48 tokenı aşan cümle |
| --- | --- | --- |
| Tweetler | 18,3 / 36 / 50 | 3 (%0,1) |
| OSCAR test | 55,4 / 136 / 955 | 967 (%40,2) |
| `oscar_dev` | 47,3 / 112 / 154 | 186 (%37,2) |

  Yani tweetlerde bu davranış skoru etkilemez; OSCAR testinde ise cümlelerin %40'ında çıktının sonu eksik kalabilir. Yayımlanmış 55,4'ün (recall 31,2) önemli bölümü bununla açıklanıyor (aşağıda, ikincil kontrol).
- **Uçtan uca hat denemesi (`oscar_dev`, 500 cümle):** P 87,2, R 80,5, F0.5 85,8 (84,1–87,3). Bu sayı bir başarı ölçüsü değildir: mT5 bu cümleleri eğitimde görmüştür. Yalnızca çıkarım → tahmin dosyası → ERRANT-TR zincirinin çalıştığını gösterir.
- **Hız (CPU, 8'li gruplar):** tweetlerde 1,25 cümle/sn (2.000 cümle 27 dakika), `oscar_dev` diliminde 0,45 cümle/sn. İki koşu aynı anda çalıştığı için tek başına daha hızlıdır.

### İkincil kontrol: OSCAR GEC testi

Aynı ayarlarla (açgözlü kod çözme, 256 token sınırı, girdi kesme yok) 2.408 cümle; çıkarım 3.128 sn (0,77 cümle/sn, CPU), skorlayıcı hiçbir cümlede hata vermedi.

| OSCAR GEC testi (2.408 cümle) | TP | FP | FN | P | R | F0.5 | %95 güven aralığı |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Yayımlanan (Ersoy ve Yıldız) | | | | 68,7 | 31,2 | 55,4 | |
| Bizim hat, `cs` | 6.890 | 3.164 | 15.476 | 68,5 | 30,8 | **55,1** | 53,7–56,5 |
| Bizim hat, `cse` | 6.888 | 3.166 | 15.478 | 68,5 | 30,8 | 55,0 | 53,7–56,5 |

**Fark 0,3 puan.** Hattımız iki farklı sette (tweetler ve OSCAR testi) yayımlanmış skoru üç değerde de (P, R, F0.5) 0,5 puanın içinde üretiyor. Tweetlerde dönüşüm uygulanıyor, OSCAR testinde uygulanmıyor ve referans resmi M2 dosyası; ikisinin birlikte tutması cetvelin doğru kurulduğunu gösteriyor.

48 token davranışı bu sette skoru belirliyor. Skor, kaynak cümlenin uzunluğuna göre ikiye ayrıldığında:

| Kaynak cümle | Cümle | TP | FP | FN | P | R | F0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 48 token ve altı | 1.441 | 3.706 | 949 | 3.776 | 79,6 | 49,5 | 71,0 |
| 48 tokenın üstü | 967 | 3.184 | 2.215 | 11.700 | 59,0 | 21,4 | 43,6 |

- 48 tokenı aşan 967 cümlenin 698'inde (%72) çıktı tam 48 tokenda bitti; 392 cümlede çıktı, kaynağın kelime sayısının %70'inden kısa. Kesilen kısımdaki bütün altın düzeltmeler kaçmış (FN), silinen kelimeler de yanlış düzeltme (FP) sayılıyor.
- **Sonuç:** yayımlanmış 55,4, modelin uzun cümleleri kesmesini de içeriyor; yazarlar da kesilmiş çıktıları skorlamış olmalı, yoksa sayılar tutmazdı. Bu sette aşılacak çıta kısa cümlelerde 71,0, uzunlarda 43,6. Uzun satırları kesmeyen ya da cümlelere bölerek işleyen bir model yalnızca bu yüzden avantaj kazanır; nihai raporda skor bu iki dilim için ayrı ayrı da verilmeli ki kazancın ne kadarının uzunluk kaynaklı olduğu görülsün.
- Kaynakla altının aynı olduğu 26 cümlenin 22'sini model değiştirdi (aşırı düzeltme oranı 0,85). Örneklem çok küçük; yalnızca gözlem.
- Çözmedeki temizleme adımı bu sette 134 çıktıyı değiştiriyor; yayımlanmış skor temizlemeli halle tutuyor, ham hal ayrıca skorlanmadı.

### İkincil kontrol: film yorumları (2026-11-24)

GECTurk M2 dosyaları indikten sonra film yorumları da bağlandı. Referans M2, kaynak ve hedef cümlelerden ERRANT-TR ile üretildi (hedef cümleler M2 düzenlemelerinin kaynağa uygulanmasıyla elde edildi; ayrıntı `01_veri_raporu.md`). Ayarlar öncekilerle aynı; dönüşüm uygulanmadı.

| Film yorumları (300 cümle) | TP | FP | FN | P | R | F0.5 | %95 güven aralığı |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Yayımlanan (Ersoy ve Yıldız) | | | | 30,0 | 22,5 | 28,1 | |
| Bizim hat, `cs` (`cse` aynı) | 53 | 120 | 174 | 30,6 | 23,3 | **28,8** | 22,6–35,3 |

**Fark 0,7 puan.** Referansta 227 düzenleme var; bu, özgün M2 dosyasındaki düzenleme sayısıyla aynı. Set küçük olduğu için güven aralığı geniş (±6 puan): bu sette iki model arasındaki birkaç puanlık fark tek başına anlamlı sayılmamalı. 150 hatasız cümlenin 20'sini mT5 değiştirdi (aşırı düzeltme oranı 0,13).

Üç setin özeti:

| Set | Yayımlanan F0.5 | Bizim hat | Fark |
| --- | --- | --- | --- |
| Tweetler (Kapı A) | 79,0 | 78,5 | −0,5 |
| OSCAR GEC testi | 55,4 | 55,1 | −0,3 |
| Film yorumları | 28,1 | 28,8 | +0,7 |

Yayımlanmış üç mT5 skorunun üçü de 1 puanın içinde yeniden üretildi. Geriye film yorumlarındaki dördüncü yayımlanmış sayı kalıyor: etiketleyicinin 84,2'si, M2 scorer ile (görev 02, adım 3 ve 4).

## Açık noktalar

- ~~Karşılaştırma modu~~: `cs` olarak sabitlendi (2026-11-24); `evaluate.py` `cse` sonucunu da raporluyor.
- **Tweet dönüşümlerinin ayrıntısı:** varsayılan ayar (`turkish`, `'’#`) yayımlanmış skoru 0,5 puan farkla üretti; ayar bu haliyle kalıyor.
- **mT5'in üretim ayarları:** yazarların değerleri hâlâ bilinmiyor, ama açgözlü kod çözme ve 256 token sınırı tweet skorunu üretiyor (`baselines.oscar_mt5`).
- **Uzun cümleler:** mT5'in 48 token davranışı OSCAR testini etkiliyor; bizim modellerimiz için uzun satırların çıkarımda nasıl ele alınacağı (cümlelere bölme) görev 03'te kararlaştırılacak.
- **Film yorumları:** ERRANT-TR tarafı bitti (28,8; yayımlanan 28,1). M2 scorer tarafı (etiketleyicinin 84,2'si) henüz bağlanmadı.
- **Süre:** OSCAR testinde gerçek bir model çıktısının değerlendirmesi 10 dakikaya yakın sürer (kaynak ve hipotez her cümlede çözümleniyor). Kaynak çözümlemesi saklanırsa kısalır; gerekirse eklenir.
- **M2 scorer** (film yorumlarındaki 84,2 için) henüz kurulmadı.
