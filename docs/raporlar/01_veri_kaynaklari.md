# Rapor 01: Veri kaynakları (görev 01, adım 1)

Tarih: 2026-11-12. Yöntem: GECTurk yerelde `datasets` ile yüklendi. Diğer kaynaklar Hugging Face API'si, GitHub API'si, depo README'leri ve Ersoy ve Yıldız (2024) makalesinin HTML sürümü üzerinden incelendi. Hiçbir veri `data/raw` altına indirilmedi.

Güven düzeyi sütunu: **Ölçüldü** = yerelde çalıştırıp saydım. **Doğrulandı** = birincil kaynakta (API, dosya, makale metni) kendim gördüm. **İkincil** = alt ajan raporu ya da tek bir kaynak; ilgili görevde yeniden bakılmalı.

## Özet: bilmen gereken beş şey

1. **GECTurk'ün Hugging Face kopyasında düzeltilmiş cümle yok.** Yalnızca `tokens` ve token başına kural numarası (`labels`) var. Bu yalnızca hata *tespiti* için yeter. Düzeltme hedefleri ve resmi skorlamada kullanılacak M2 dosyaları yalnızca GECTurk README'sindeki **Google Drive** klasöründe. Bu dosyaların senin tarafından indirilmesi gerekiyor.
2. **OSCAR test setindeki 62,8, OSCAR GEC (mT5) modeline değil, GPT GEC (mT5) modeline ait.** Tweetlerde 79,0 alan OSCAR GEC (mT5) modeli OSCAR testinde **55,4** alıyor. Proje özetindeki tablo iki modeli karıştırmış.
3. **Ersoy ve Yıldız üç test setinin hepsini ERRANT-TR ile skorlamış**, M2 scorer ile değil. Film yorumlarında etiketleyicinin 84,2'si ise GECTurk makalesinden, GECTurk'ün M2 scorer'ıyla. Yani film yorumlarındaki iki yayımlanmış sayı farklı araçlardan geliyor (K15'i etkiler).
4. **Üç kaynağın lisansı yok ya da belirsiz:** tweet seti, Turkish-GEC GitHub deposu ve mT5 model ağırlıkları. Görev tanımı gereği burada durup soruyorum.
5. **Kumru'nun Hugging Face kimliği değişmiş:** `vngrs-ai/Kumru-2B` artık `vngrs/Kumru-2B`'ye yönleniyor ve model 2,38 milyar parametre (K6'daki 1–2 milyar aralığının biraz üstünde).

## 1. GECTurk

| Özellik | Bulgu | Güven |
| --- | --- | --- |
| HF kimliği | `GGLab/GECTurk`, tek yapılandırma (`default`), lisans Apache-2.0 | Ölçüldü |
| Ayrımlar | train 96.919, dev 20.769, test 20.769, movie_reviews 300 | Ölçüldü |
| Sütunlar | `tokens: list[str]`, `labels: list[int64]` (token başına). **Hedef cümle yok** | Ölçüldü |
| Etiket numaraları | `0 = O` (hata yok), `k = rule_k` (1–25). `GGLab/gec-tr-seq-tagger` `config.json` içindeki `id2label` ile aynı sıra | Doğrulandı |
| Kural adları | `rule_1` … `rule_25`. Adlar (ör. 1 CONJ_DE_SEP, 6 CASE_DE, 23 SENT_CAP) GECTurk makalesinin Tablo 1'inden; 10, 11, 19, 21'in adları makalede yok | İkincil |
| M2 dosyaları | GitHub deposunda **yok**. README: veri Drive klasöründe (`drive.google.com/drive/folders/121DNJ65WiCl3LvgDc9Cb9EXtNRk6XThR`). Film yorumları ayrıca tek dosya olarak Drive'da | Doğrulandı |
| M2 biçimi | `A 6 8\|\|\|rule_20\|\|\|olabilir\|\|\|REQUIRED\|\|\|-NONE-\|\|\|0`: kural etiketi M2 düzenleme satırında | Doğrulandı (README) |
| M2 scorer | Depoda `utils/m2_scorer/m2scorer.py`, NUS M2 scorer'ın Python 3 sürümü, GPL v3 | İkincil |
| Etiketleyici | `GGLab/gec-tr-seq-tagger` var; BertForTokenClassification, 26 etiket, transformers 4.25.1, Apache-2.0. Depoda bağımsız çıkarım betiği ve sabitlenmiş sürüm yok | Doğrulandı (config) / İkincil (kod) |
| GitHub deposu lisansı | Yok (`license: null`). HF veri seti ve model Apache-2.0 | İkincil |
| Film yorumları kaynağı | beyazperde.com, Altınok (2023) paylaşmış, bir uzman etiketlemiş | İkincil |

### Ölçülen istatistikler

| Ayrım | Cümle | Hatasız | Ort. uzunluk (token) | En uzun | Birden çok kural içeren |
| --- | --- | --- | --- | --- | --- |
| train | 96.919 | %50,3 | 11,2 | 117 | 14.104 |
| dev | 20.769 | %50,6 | 11,2 | 153 | 3.009 |
| test | 20.769 | %50,3 | 11,1 | 81 | 3.082 |
| movie_reviews | 300 | %50,0 | 15,0 | 58 | 49 |

Train'de kuralların geçtiği cümle sayısı (en sıktan en nadire): 6: 21.875, 1: 8.592, 12: 8.033, 5: 7.261, 20: 5.110, 11: 3.681, 22: 2.525, 23: 1.828, 7: 1.251, 13: 1.086, 8: 976, 14: 605, 24: 583, 18: 396, 4: 330, 17: 322, 16: 286, 10: 266, 15: 228, 9: 197, 25: 193, 2: 72, 3: 66, 21: 38, **19: 3**.

**Örnekleme için anlamı (K10):** 19 ve 21 numaralı kurallar 50'nin altında; `min_per_rule=50` bunların hepsini alır. 19 numaralı kuraldan train'de yalnızca 3 cümle var; model bu kuralı fiilen öğrenemez. Raporda sınırlılık olarak yazılmalı.

### Tokenizasyon gözlemi (adım 2'nin ön bulgusu)

GECTurk'te noktalama kelimeye **bitişik**: `başlıyor.`, `okul,`, `"çok`. Train'deki 1.082.934 tokenın yalnızca 1.781'i tek başına noktalama. Ayrıca kesme işareti de bitişik (`Başbakan'ın`).

## 2. OSCAR GEC (Ersoy ve Yıldız 2024)

| Özellik | Bulgu | Güven |
| --- | --- | --- |
| Eğitim verisi | `asimokby/Turkish-OSCAR-GEC`, `train.csv` + `val.csv`; sütunlar `sentence`, `correction` (paralel); hata türü etiketi yok | İkincil |
| Satır sayısı | train 1.737.696, validation 579.233 (toplam 2.316.929; makale 2.326.921 diyor) | İkincil |
| Lisans | MIT (HF veri kartı) | İkincil |
| Test seti | `asimokby/Turkish-GEC-Evaluation`: `inp.txt`, `ref.txt` ve `oscar_gec_m2_W_MAPPED_LABELS.txt` (ERRANT-TR etiketli M2). Lisans MIT | İkincil |
| Test seti boyutu | 2.408 cümle; makaleye göre eğitim ve doğrulamadan **çıkarılmış** | Doğrulandı (makale) |
| Test tokenizasyonu | M2'de noktalama **ayrı token**: `resim gönderdim , Gelmediyse` | İkincil |
| mT5 modelleri | `asimokby/Turkish-OSCAR-GEC-v0` ve `asimokby/Turkish-GPT-GEC-v0`, ikisi de mT5-base (582M). **Model kartı ve lisans yok.** Yüklenen checkpoint'in makalede skorlanan model olduğu yazılı değil | İkincil |
| GitHub deposu | Yalnızca README; kod ve skorlama betiği yok; lisans yok | İkincil |

## 3. #Turki$hTweets

| Özellik | Bulgu | Güven |
| --- | --- | --- |
| Konum | `github.com/atubakoksal/annotated_tweets`, tek veri dosyası `data.txt` | Doğrulandı |
| Boyut | 2.000 tweet + başlık satırı. Ersoy ve Yıldız 1.996 diyor | Ölçüldü (satır sayısı) |
| Biçim | Sekmeyle ayrılmış: ham, ön işlenmiş, düzeltilmiş, etiketler. Başlıkta fazladan bir sekme ve UTF-8 BOM var | Ölçüldü |
| Etiketler | Düzeltilmiş cümlede kelime başına bir etiket: `IV` + 13 OOV türü (deascii, int, accent, proper, unint, sep, joint, vowel, phonetic_sub, foreign, social, abbr, neologism) | İkincil |
| Tokenizasyon | Düzeltilmiş sütun küçük harfli ve noktalamasız | İkincil |
| Train/test ayrımı | Yok; tamamı test olarak kullanılıyor | İkincil |
| Lisans | **Yok.** README yalnızca başlık ve kaynakça | Doğrulandı |
| Ersoy ve Yıldız'ın sürümü | Kendi kopyalarını **yayımlamamışlar**. Makale iki dönüşüm uyguladıklarını söylüyor: altın cümlelerin ilk harfini büyütmek ve model çıktılarındaki noktalamayı silmek | Doğrulandı (makale) |

## 4. Lisans özeti

| Kaynak | Lisans | Kullanım için sorun? |
| --- | --- | --- |
| GECTurk (HF veri ve etiketleyici) | Apache-2.0 | Yok |
| GECTurk GitHub deposu (M2 scorer dahil) | Depo lisansı yok; M2 scorer dosyası GPL v3 | Skorlayıcıyı çalıştırmak sorun değil; koda kopyalarsak GPL uyarısı gerekir |
| Turkish-OSCAR-GEC, Turkish-GEC-Evaluation | MIT | Yok |
| Turkish-OSCAR-GEC-v0 / GPT-GEC-v0 (mT5) | **Belirtilmemiş** | Akademik değerlendirme için çalıştırmak yaygın; danışmana sorulmalı |
| #Turki$hTweets | **Belirtilmemiş** | Yalnızca değerlendirmede kullanılacak, yeniden dağıtılmayacak; danışmana sorulmalı |
| ERRANT-TR | MIT | Yok |
| Kumru-2B, Qwen3-1.7B | Apache-2.0 | Yok |

## 5. Senin kararını ya da işlemini bekleyenler

1. **GECTurk M2 dosyalarını Drive'dan indir.** Drive klasörü komut satırından güvenilir biçimde indirilemiyor. Klasörü tarayıcıda açıp `gec_tr/` alt klasörünü ve film yorumları dosyasını `data/raw/gecturk/` altına koyman gerekiyor. Bu olmadan görev 01'in 3. adımı (ortak biçime çevirme) başlayamaz.
2. **Lisansı olmayan kaynaklar** (tweetler, mT5 ağırlıkları): yalnızca değerlendirme için ve veriyi yeniden dağıtmadan kullanmayı danışmanla teyit etmeni öneririm.
3. **OSCAR testindeki hedef sayı:** temel model OSCAR GEC (mT5) ise doğru karşılaştırma **55,4**. 62,8 (GPT GEC mT5) ikinci, daha güçlü bir referans olarak eklenebilir.
4. **Film yorumlarında skorlayıcı (K15):** 84,2 M2 scorer'dan, 28,1 ERRANT-TR'den geliyor. Önerim: film yorumlarında iki aracı da çalıştırıp her sayıyı kendi aracıyla kıyaslamak.

Varsayımların tek tek sonuçları: `docs/raporlar/00_varsayim_dogrulama.md`.
