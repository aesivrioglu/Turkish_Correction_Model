# Rapor 01b: Tokenizasyon incelemesi (görev 01, adım 2)

Tarih: 2026-11-13. Test setlerinde yalnızca biçim incelendi (satır sayısı, tokenizasyon, harf durumu, hatasız oran); içerikten karar çıkarılmadı.

## İndirilen dosyalar

Bu dosyalar `data/raw/` altında duruyor ve depoya girmiyor. Sürümler sabitlendi.

| Set | Kaynak ve sürüm | Dosyalar | SHA-256 (ilk 12) |
| --- | --- | --- | --- |
| OSCAR GEC test | HF `asimokby/Turkish-GEC-Evaluation` @ `04cd0c3` | `inp.txt`, `ref.txt` (2.408 satır), `oscar_gec_m2_W_MAPPED_LABELS.txt` (2.408 blok) | `36c2a8994b8e`, `cc7a174606d3`, `37263bf6da06` |
| #Turki$hTweets | GitHub `atubakoksal/annotated_tweets` @ `06e9476` | `data.txt` (2.000 satır + başlık) | `0219f29df660` |
| GECTurk | HF `GGLab/GECTurk` (parquet, önbellekte) | M2 dosyaları **henüz yok** (Drive) | |

## Bulgular

### GECTurk (parquet)

- `tokens`, doğal metnin boşlukla bölünmüş halidir. Noktalama ve kesme işareti kelimeye bitişiktir (`başlıyor.`, `Başbakan'ın`). 1.082.934 tokenın yalnızca %0,16'sı tek başına noktalama.
- Yani `" ".join(tokens)` doğal metni verir.
- M2 scorer hipotezi boşluktan böler. Bu yüzden modelin doğal metin çıktısı, kaynakla aynı boşluk düzenini koruduğu sürece doğrudan skorlanabilir.
- **Açık kalan:** Drive'daki M2 dosyalarının `S` satırının parquet tokenlarıyla birebir aynı olduğu, M2 dosyaları gelince doğrulanacak.

### OSCAR GEC test

| Dosya | Biçim |
| --- | --- |
| `inp.txt`, `ref.txt` | Doğal metin, noktalama bitişik (`gönderdim,Gelmediyse`) |
| M2 `S` satırı | ERRANT-TR'nin (Zemberek) tokenize ettiği biçim, noktalama ayrı (`gönderdim , Gelmediyse`). 2.408 bloğun yalnızca 291'inde `inp` ile aynı |

- Satırlar çoğu zaman birden çok cümle içeriyor: ortalama 25 token.
- Hatasız satır yalnızca 26 (%1,1). Bu sette aşırı düzeltme oranı anlamlı ölçülemez; o ölçüm GECTurk dev ve film yorumlarında yapılır.
- Tek annotator (kimlik `0`); toplam 22.392 düzenleme.

### #Turki$hTweets

| Sütun | Biçim |
| --- | --- |
| `RAW DATA` | Ham tweet: emoji, ifade (`:)`), büyük harf |
| `PREPROCESSED DATA` | Büyük ölçüde küçük harfli ve noktalamasız. Kalanlar: kesme işareti, `#`, `\|` |
| `CORRECTED DATA` | Özel isimler büyük (`İsmail`), kesme işareti var (`Adana'da`). İlk harfi büyük olan yalnızca 229/2.000 |
| `ANNOTATED DATA` | Kelime başına etiket, `&` ile birleşik. 2.000 satırın hepsinde etiket sayısı kelime sayısına eşit |

- **`|` işareti boşluk demek.** İki sütunda da kelimeleri bire bir hizalı tutmak için kullanılmış: `tabiki → tabii|ki` (sep), `bir|çok → birçok` (joint). Doğal metne çevirirken `|` boşluğa dönmeli.
- **Kaynak sütunda 47 tane birleşen nokta (U+0307) var** (`beri̇`, `yedi̇m`). Bu, veri setinin Python `str.lower()` ile küçültülmesinden kalan bir iz. Ölçütün kaynak metnini değiştirmiyoruz ama raporda belirtilmeli.
- Hatasız tweet: 89/2.000 (%4,5).
- Dosyada UTF-8 BOM ve başlıkta fazladan bir sekme var.

### ERRANT-TR'nin girdisi

- `erranttr/process.py` (commit `56cf0a3`), her satırı Zemberek'in `analyze_and_disambiguate` işleviyle çözümlüyor. Tokenizasyonu kendisi yapıyor. Yani **girdi doğal metindir**; önceden tokenize etmeye gerek yok.
- `commands/parallel_to_m2.py`, İngilizce ERRANT'tan kopyalanmış ve Türkçe sürüme uyarlanmamış görünüyor (`token.text`, `args.tok`). Görev 02'de `process.py` yolu kullanılmalı.

## Karar önerisi (onayını bekliyor)

**Ortak JSONL biçiminde doğal metin saklanır; hiçbir sete tokenizasyon uygulanmaz.** Her skorlayıcı kendi tokenizasyonunu yapar:

| Set | `source` | `target` | Skorlayıcıya giden |
| --- | --- | --- | --- |
| GECTurk (train, dev, test, film yorumları) | `" ".join(tokens)` | M2'den uygulanmış düzeltme | M2 scorer: özgün M2 + hipotez (boşlukla bölünür). Film yorumlarında ayrıca ERRANT-TR |
| OSCAR test | `inp.txt` satırı | `ref.txt` satırı | ERRANT-TR: hipotezden M2 üretilir, özgün M2 ile kıyaslanır |
| Tweetler | `PREPROCESSED`, `\|` → boşluk | `CORRECTED`, `\|` → boşluk | ERRANT-TR. Ersoy ve Yıldız'ın iki dönüşümü değerlendirme katmanında uygulanır, JSONL'de değil |

Gerekçeler:

1. Model gerçek kullanımda doğal metin görecek; demo da doğal metinle çalışacak.
2. Skorlayıcıların beklediği tokenizasyon bozulmaz: GECTurk'ün doğal metni zaten M2 scorer'ın beklediği biçimde, ERRANT-TR de kendi tokenizasyonunu yapıyor.
3. Ölçüte özgü dönüşümler (tweetlerdeki büyük harf ve noktalama) tek bir yerde, değerlendirmede durur. Böylece hangi sayının hangi dönüşümle üretildiği açık kalır.

Özgün etiketler `extra` alanında saklanır: tweet etiketleri ve OSCAR M2 düzenleme türleri.

## Kapı A için not

Ersoy ve Yıldız, tweet setinde "model çıktılarındaki noktalamayı sildiklerini" yazıyor. Ama altın cümlelerde 367 kesme işareti var (`Adana'da`). Bütün noktalama silinseydi her özel isim eki hata sayılırdı. Demek ki büyük olasılıkla yalnızca cümle noktalaması (`. , ! ?` gibi) silinmiş. Kapı A'da 79,0 tutmazsa ilk denenecek değişken bu.
