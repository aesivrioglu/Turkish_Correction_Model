# Rapor 03a: Model adayları ve beklenen eğitim süreleri

Tarih: 2026-11-21. Görev 03, adım 1'in genişletilmiş hali. Danışmanın "denenebilecek model sayısı artırılabilir" önerisi üzerine hazırlandı. Bu rapor bir karar değildir; adayları birlikte değerlendirmek için hazırlanmış bir listedir. Hiçbir yapılandırma ya da karar değiştirilmedi.

Güven düzeyi: **Ölçüldü** = yerelde çalıştırıp saydım. **Doğrulandı** = birincil kaynakta (HF API, model kartı) kendim gördüm. **İkincil** = alt ajan araştırması; kaynağı aşağıda, ilgili görevde yeniden bakılmalı. **Tahmin** = ölçülmüş verilerden türetilmiş hesap.

## Kısa sonuç

1. **Mevcut iki adayda D1 tek T4'te 3 saatin altında, D2 değil.** Kumru-2B için D1 yaklaşık 1,8 saat, D2 yaklaşık 3,7 saat. Qwen3-1.7B için D1 yaklaşık 2,2 saat, D2 yaklaşık 4,7 saat (tahmin; aralık bölüm 1.3'te). İki T4'ü birlikte kullanınca (DDP) D2 yaklaşık 2,0 ve 2,6 saate iniyor. Bu sayılar iş planındaki "D1 ve D2 için 5–11 GPU saati" tahminiyle tutarlı.
2. **Kumru, parametresi daha çok olduğu halde Qwen3-1.7B'den hızlı.** Türkçe tokenizer'ı aynı cümleyi daha az tokenla ifade ediyor: kelime başına 1,81 token, Qwen3'te 2,74. Bir eğitim örneği Kumru'da ortalama 136 token, Qwen3'te 186 (ölçüldü).
3. **"En fazla 3 saat" sınırı için iki T4'ü birlikte kullanmak gerekiyor.** Tek T4'te D2 yalnızca 0,8B'ye kadar olan modellerde 3 saatin altında kalıyor. İki T4'le yaklaşık 2,4B'ye kadar sığıyor. 3–4B modellerde D2 iki T4'le bile 3,5–4,7 saat, 7–9B modellerde 5,5–8 saat.
4. **Mini eğitimler (2 bin örnek) ucuz.** 2,4B'ye kadar her aday 25 dakikanın altında. Altı adaylık bir pilot, yükleme ve değerlendirmeyle birlikte yaklaşık 2,5–3 GPU saati tutuyor. İki T4'te iki aday aynı anda eğitilirse duvar saatiyle yaklaşık 1,5 saat.
5. **Önerim:** pilotu iki adaydan altıya çıkaralım, D1 ve D2'yi yine tek modelle yapalım. D2'yi 3 saatin altına indirmek için iki T4'ü DDP ile kullanalım. İkinci bir modelle tam deneyi pilot sonucuna göre konuşalım (bölüm 6 ve 7).

## 1. Süre nasıl hesaplandı?

### 1.1 Örnek uzunlukları (ölçüldü)

GECTurk train'den rastgele 3.000 cümle (tohum 42) ve OSCAR GEC train'in ilk 3.000 satırı üzerinde ölçüldü. Yalnızca eğitim verisi kullanıldı; test setlerine dokunulmadı. Örnekler `trgec.formats.build_messages` ile kuruldu, sonra her modelin kendi `apply_chat_template` işlevinden geçirildi.

**Sınırlılık:** GECTurk'ün HF kopyasında hedef cümle yok (M2 dosyaları hâlâ bekleniyor). Bu yüzden GECTurk örneklerinde hedef yaklaşık kuruldu: düzeltilmiş cümle yerine kaynak cümle, her hatalı token için bir düzeltme satırı. Kesin `max_len` ölçümü görev 03'te gerçek hedeflerle yapılacak. Bu ölçüm proje bağımlılıklarına dokunmadan, geçici bir sanal ortamda (transformers 5.23.0, yalnızca tokenizer) yapıldı.

| Tokenizer (aynı tokenizer'ı kullanan modeller) | Sözlük | Kelime başına token | GECTurk örneği ort. / %95 | OSCAR örneği ort. / %95 |
| --- | --- | --- | --- | --- |
| `ufakai/ufakzeka-1` | 40.960 | 1,72 | 130 / 207 | 139 / 290 |
| `ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1` (ve medium) | 50.258 | 1,68 | 135 / 213 | 133 / 270 |
| `boun-tabi-LMG/TURNA` | 32.100 | 1,70 | 132 / 208 | 131 / 275 |
| `asafaya/kanarya-750m` (ve 2b) | 32.768 | 1,72 | 142 / 221 | 138 / 282 |
| `vngrs/Kumru-2B` (ve Kara-Kumru) | 50.176 | 1,81 | 136 / 213 | 140 / 293 |
| `TURKCELL/Turkcell-LLM-7b-v1` | 48.353 | 1,98 | 152 / 236 | 159 / 327 |
| `utter-project/EuroLLM-1.7B-Instruct` | 128.000 | 2,37 | 187 / 283 | 198 / 393 |
| Llama-3 (Llama-3.2, Cosmos Turkish-Llama-8b, Cere-Llama) | 128.256 | 2,41 | 172 / 260 * | 192 / 382 * |
| `microsoft/Phi-4-mini-instruct` | 200.029 | 2,41 | 169 / 260 | 182 / 371 |
| `Qwen/Qwen3.5-2B` (ve cisimcik-4b) | 248.077 | 2,47 | 180 / 271 | 196 / 388 |
| Qwen3 (0.6B, 1.7B, 4B, 8B, Mecellem, Trendyol-8B-T1) | 151.669 | 2,74 | 186 / 286 | 212 / 432 |
| `BSC-LT/salamandra-2b-instruct` | 256.000 | 3,64 | 420 / 560 | 433 / 725 |

\* Llama-3.2'nin şablonu tarih içeren bir sistem mesajı ekliyor; Llama-3.2-1B'de ortalama 197 / 217. Cosmos şablonunda bu yok.

Şablonla ilgili iki gözlem:

- `mistralai/Ministral-3-3B-Instruct-2512`'nin varsayılan sistem mesajı tek başına yaklaşık 650 token. Bu model seçilseydi sistem mesajı değiştirilmeliydi.
- Qwen3'te `enable_thinking=False` cevaba boş bir `<think>` bloğu ekliyor (yaklaşık 4 token).

### 1.2 Dolgu israfı (ölçüldü)

Aynı 6.000 örnekte Qwen3 tokenizer'ıyla batch'leri simüle ettim:

| Batch boyutu | Rastgele batch | Uzunluğa göre gruplu |
| --- | --- | --- |
| 8 | %95 | %16 |
| 16 | %152 | %25 |
| 32 | %243 | %29 |

OSCAR örneklerinin uzun bir kuyruğu var (ortalama 212, %95'lik dilim 432). Rastgele batch'te hemen her batch'e uzun bir örnek düşüyor ve geri kalanlar ona kadar dolduruluyor. **Uzunluğa göre gruplama zorunlu.** Bu bir karar değişikliği değil, `train.py` için bir uygulama notu. Hesapta dolgu payı %20 alındı.

### 1.3 Eğitim hızı (ikincil ve tahmin)

T4 için iki ölçülmüş çapa kullandım (ikincil, kaynaklar aşağıda):

- **Çapa 1:** Qwen2.5-0.5B (0,49B), HF + 16-bit LoRA, tek T4: saniyede 1.365 token. QLoRA'nın küçük modellerde 16-bit LoRA'ya göre yaklaşık 0,73 kat hızda çalıştığı varsayıldı (V100 ölçümü, 0,71–0,76). Sonuç: saniyede yaklaşık 1.000 token.
- **Çapa 2:** yaklaşık 7B, HF + QLoRA tabanı (Unsloth'un karşılaştırma tablosu): örnek başına 1,62 saniye, yani saniyede yaklaşık 135 token.

Bu iki noktadan geçen bir üs yasası:

```
token/s(N) ≈ 581 × N^(-0,764)      N: milyar parametre
süre = (örnek sayısı × ortalama uzunluk × 1,2 dolgu) / token/s
```

Hesaba katılmayan iki ölçümle kontrol ettim:

| Ölçüm | Ölçülen | Formül |
| --- | --- | --- |
| Qwen2.5-1.5B, HF QLoRA | saniyede 2,7 örnek, yaklaşık 405 token (örnek uzunluğu tahmini) | 418 |
| Gemma-2-2B (2,6B), Unsloth QLoRA, HF'ye çevrilmiş (÷1,7) | yaklaşık 224 token | 280 |

**Belirsizlik büyük.** Gerçek süre, tablodaki merkez değerin 0,6 ile 1,5 katı arasında olabilir. Çapalardaki mikro-batch 2–4. Bizim kısa örneklerimizle 16'lık mikro-batch GPU'yu daha iyi doldurur; bu yüzden gerçek hız daha yüksek çıkabilir. Kesin hız pilotta ölçülecek (görev 03, adım 6).

**Not:** Bu raporun ilk taslağında çapa 1'i yanlışlıkla 0,75B'ye ve QLoRA'ya bağlamıştım. Bu, süreleri yaklaşık 1,5 kat kısa gösteriyordu. Yukarıdaki sayılar düzeltilmiş hesaptan.

### 1.4 Varsayımlar

- HF transformers + PEFT, QLoRA (4-bit NF4, fp16), gradient checkpointing açık, 1 epok.
- Mini: 1 bin GECTurk + 1 bin OSCAR. D1: 12 bin GECTurk. D2: 12 bin GECTurk + 12 bin OSCAR.
- Model indirme ve yükleme için ayrıca 5–15 dakika eklenmeli. Çıkarım süresi bölüm 4'te.
- "2×T4" sütunu, tek eğitimin iki T4'te DDP ile koştuğu ve 1,8 kat hızlandığı varsayımıyla hesaplandı (tahmin; T4'te bağımsız ölçüm bulunamadı).

## 2. Mevcut iki aday

| Model | Parametre | Mini | D1 | D2 | D2 aralığı | D2, 2×T4 |
| --- | --- | --- | --- | --- | --- | --- |
| `vngrs/Kumru-2B` | 2,38B | 18 dk | 1,8 sa | 3,7 sa | 2,3–5,7 sa | 2,0 sa |
| `Qwen/Qwen3-1.7B` | 2,03B | 24 dk | 2,2 sa | 4,7 sa | 2,9–7,2 sa | 2,6 sa |

Kimlik, lisans (ikisi de Apache-2.0) ve parametre sayısı yeniden doğrulandı (HF API, 2026-11-21). İkisi de gated değil ve saf transformer.

## 3. Yeni adaylar

Bütün satırlarda kimlik, parametre sayısı ve lisans HF API ile doğrulandı. Türkçe desteği model kartından okundu; aksi belirtilmişse ikincil. Süreler tahmindir; "2×T4" dışındaki sütunlar tek T4 içindir.

### 3.1 Kısa süren (D2 tek T4'te yaklaşık 2 saat ya da daha az)

| Model | Parametre | Lisans | Mini | D1 | D2 | D2, 2×T4 | Not |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ufakai/ufakzeka-1` | 0,18B | Apache-2.0 | 2 dk | 14 dk | 30 dk | 17 dk | Türkçe, sıfırdan eğitilmiş, Qwen3 mimarisi, Eylül 2026. Çok yeni ve çok küçük; ancak "alt sınır" referansı olabilir |
| `ytu-ce-cosmos/turkish-gpt2-medium-350m-instruct-v0.1` | 0,35B | MIT | 4 dk | 25 dk | 50 dk | 28 dk | Türkçe GPT-2, şablonlu |
| `ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1` | 0,77B | MIT | 8 dk | 46 dk | 1,5 sa | 50 dk | Türkçe GPT-2, şablonlu. Bağlam 1.024 token; bizim %95'lik dilimimize (≤ 300) yetiyor |
| `asafaya/kanarya-750m` | 0,74B | Apache-2.0 | 8 dk | 47 dk | 1,5 sa | 51 dk | Türkçe GPT-J. Base model, şablon yok; şablonu bizim tanımlamamız gerekir |
| `Qwen/Qwen3-0.6B` | 0,75B | Apache-2.0 | 11 dk | 1,0 sa | 2,2 sa | 1,2 sa | Qwen3-1.7B ile aynı aile ve tokenizer. Boyut etkisini temiz gösterir |
| `boun-tabi-LMG/TURNA` | 1,14B | **Yalnızca ticari olmayan akademik kullanım** | ~6 dk | ~35 dk | ~1,2 sa | ~39 dk | Türkçe encoder-decoder (T5 ailesi). Decoder değil, K6'nın dışında. T5 ailesinde fp16 taşması bilinen bir sorun. Süre kaba: encoder ve decoder yarı yarıya sayıldı |

### 3.2 Hedef bant (D1 tek T4'te ≤ 2,2 saat; D2 iki T4'te ≤ 3 saat)

| Model | Parametre | Lisans | Mini | D1 | D2 | D2, 2×T4 | Not |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `meta-llama/Llama-3.2-1B-Instruct` | 1,24B | Llama 3.2 (gated) | 17 dk | 1,6 sa | 3,4 sa | 1,9 sa | Türkçe resmi listede **yok** |
| `utter-project/EuroLLM-1.7B-Instruct` | 1,66B | Apache-2.0 | 20 dk | 1,9 sa | 3,9 sa | 2,2 sa | Türkçe resmi listede **var** (kartta `tr`, doğrulandı). Üçüncü bir aile |
| `newmindai/Mecellem-Qwen3-1.7B-TR` | 2,03B | Apache-2.0 | 24 dk | 2,2 sa | 4,7 sa | 2,6 sa | Qwen3-1.7B üzerine yaklaşık 225B token Türkçe sürekli ön eğitim, hukuk ağırlıklı (ikincil). Base model. Qwen3-1.7B ile aynı mimari ve tokenizer; fark yalnızca Türkçe ön eğitim |
| `asafaya/kanarya-2b` | 2,06B | Apache-2.0 | 17 dk | 1,7 sa | 3,3 sa | 1,9 sa | Türkçe GPT-J, base, şablon yok, bağlam 2.048 |
| `AlicanKiraz0/Kara-Kumru-v1.0-2B` | 2,38B | Apache-2.0 | 18 dk | 1,8 sa | 3,7 sa | 2,0 sa | Kumru-2B üzerine topluluk SFT'si. Kumru'dan ayrı bir şey ölçmez |

### 3.3 Uzun süren, 3–4B (D2 iki T4'te bile 3,5–4,7 saat)

| Model | Parametre | Lisans | Mini | D1 | D2 | D2, 2×T4 | Not |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `meta-llama/Llama-3.2-3B-Instruct` | 3,21B | Llama 3.2 (gated) | 35 dk | 3,3 sa | 6,9 sa | 3,9 sa | Türkçe resmi listede yok |
| `CohereLabs/tiny-aya-global` | 3,35B | **CC-BY-NC-4.0** (gated) | ~32 dk | ~3,0 sa | ~6,3 sa | ~3,5 sa | Türkçe resmi (ikincil). Tokenizer gated olduğu için ölçülemedi; token sayısı tahmin. transformers v5 gerekebilir |
| `microsoft/Phi-4-mini-instruct` | 3,84B | MIT | 34 dk | 3,3 sa | 6,8 sa | 3,8 sa | Türkçe resmi (kartta `tr`, doğrulandı) |
| `Qwen/Qwen3-4B-Instruct-2507` | 4,02B | Apache-2.0 | 40 dk | 3,7 sa | 7,9 sa | 4,4 sa | Qwen3 ailesinin bir üst basamağı |
| `newmindai/Mecellem-Qwen3-4B-TR` | 4,41B | Apache-2.0 | 43 dk | 4,0 sa | 8,5 sa | 4,7 sa | Türkçe sürekli ön eğitimli Qwen3-4B, base |

### 3.4 Uzun süren, 7–9B (D2 tek T4'te 10–14 saat)

| Model | Parametre | Lisans | Mini | D1 | D2 | D2, 2×T4 | Not |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `TURKCELL/Turkcell-LLM-7b-v1` | 7,38B | Apache-2.0 | 49 dk | 4,8 sa | 9,9 sa | 5,5 sa | Mistral tabanlı, genişletilmiş Türkçe tokenizer |
| `ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1` | 8,03B | Llama 3 | 1,0 sa | 5,8 sa | 12,3 sa | 6,8 sa | 30 GB Türkçe veriyle tam ince ayar (ikincil) |
| `CerebrumTech/cere-llama-3-8b-tr` | 8,03B | Llama 3 | ~1,0 sa | ~5,8 sa | ~12,3 sa | ~6,8 sa | Cetvel'in GEC görevinde en yüksek resmi skor, 46,0 (ikincil, bkz. bölüm 8) |
| `Qwen/Qwen3-8B` | 8,19B | Apache-2.0 | 1,1 sa | 6,4 sa | 13,7 sa | 7,6 sa | |
| `Trendyol/Trendyol-LLM-8B-T1` | 8,19B | Apache-2.0 | 1,1 sa | 6,4 sa | 13,7 sa | 7,6 sa | Qwen3-8B tabanlı, Türkçe; akıl yürütme modeli |
| `utter-project/EuroLLM-9B-Instruct-2512` | 9,15B | Apache-2.0 | 1,2 sa | 7,0 sa | 14,4 sa | 8,0 sa | Türkçe resmi (ikincil) |

Bu boyutta D2 tek T4'te 12 saatlik oturum sınırını zorluyor ya da aşıyor. Bu modeller ancak tek bir "üst referans" olarak işe yarar: ya yalnızca mini eğitim ya da küçültülmüş veriyle tek bir eğitim. `CohereLabs/aya-expanse-8b` de Türkçe resmi, ama CC-BY-NC lisanslı ve gated.

### 3.5 Elenenler

| Model ya da aile | Neden |
| --- | --- |
| Gemma ailesi ve türevleri (Turkish-Gemma-9b, Turkish-Gemma-4b, Trendyol Asure-12B, WiroAI, Tower-Plus) | K12 |
| Qwen3.5 küçük modelleri ve türevleri (`cisimcik/cisimcik-4b`) | Gated DeltaNet hibriti. Hızlı çekirdekler Triton tabanlı; Triton resmi olarak compute capability ≥ 8.0 istiyor, T4 7.5. Unsloth bu ailede QLoRA önermiyor. cisimcik'in SFT verisinde makine üretimi GEC örnekleri var; GECTurk kökenli bir sızıntı riski taşıyor (ikincil) |
| Ministral 3 (3B, 8B) | Instruct ağırlıkları FP8, T4 desteklemiyor. Türkçe resmi listede yok. Varsayılan sistem mesajı yaklaşık 650 token (ölçüldü) |
| Salamandra | Türkçe yok; kelime başına 3,64 token (ölçüldü), en verimsiz tokenizer |
| Granite 4.x, LFM2 / LFM2.5 / LFM3, SmolLM3, Falcon-H1, OLMo, Nemotron, Phi-4-mini-flash | Türkçe resmi listede yok ve/veya Mamba ya da konvolüsyon hibriti (ikincil) |
| Kumru-7B, T3 AI, TÜBİTAK BİLGE | Açık ağırlık yok (Kumru-7B için HF API 401 döndü) |
| `UgurI/Qwen3.8-27B-Turkish-CPT-Beta`, `magibu/magibu-11b-v0.8` | Boyut dışı, lisans sorunlu ya da gated |

## 4. Çıkarım süresi ve Kaggle bütçesi

**Çıkarım (tahmin).** Her eğitimden sonra doğrulama setleri ve en sonda üç test seti (toplam yaklaşık 5.700 cümle) için tahmin üretilecek. Açgözlü üretim ve batch'li çıkarımla kaba tahminler:

| Boyut | 5.700 cümle |
| --- | --- |
| ≤ 2,4B | 15–30 dk |
| 3–4B | 30–60 dk |
| 7–9B | 1–2 sa |

LoRA'yı birleştirip fp16 üretmek, 2B modellerde 4-bit üretimden hızlı olabilir. 8B'nin fp16 ağırlıkları (yaklaşık 16 GB) tek T4'e sığmaz.

**Kaggle (ikincil, Kasım 2026).**

- Haftada yaklaşık 30 GPU saati; oturum başına 12 saat.
- 15 Eylül 2026'dan beri ücretsiz GPU seçeneği yalnızca T4×2 (P100 kaldırıldı).
- 4 CPU, 29 GB RAM, `/kaggle/working` altında 20 GB kalıcı disk.
- İkincil bir kaynağa göre kota GPU sayısına değil oturum süresine göre düşüyor. Öyleyse iki T4'ü kullanmak kotayı iki kat harcamıyor. **Doğrulanmadı;** hesap ayarlarındaki kota ekranından kontrol edilmeli.

## 5. Süreyi kısaltan seçenekler

| Seçenek | Beklenen kazanç | Bedel | Karar gerekir mi? |
| --- | --- | --- | --- |
| Uzunluğa göre gruplama | Dolgu %95–240'tan %16–29'a iner (ölçüldü) | Yok | Hayır, uygulama ayrıntısı |
| Tek eğitimde 2×T4 DDP (`accelerate`, zaten Kaggle gereksinimlerinde) | Yaklaşık 1,7–1,9 kat (tahmin). Mevcut adaylarda D2'yi 3 saatin altına indiren yol | Kaggle'da not defteri içinden başlatma sorunlu; betik `.py` olarak `accelerate launch` ile başlatılmalı. Her GPU 4-bit modelin bir kopyasını tutar | Hayır, yeni bağımlılık yok |
| İki T4'te iki ayrı eğitim (`CUDA_VISIBLE_DEVICES=0` ve `=1`) | Pilotta iki aday aynı anda eğitilir | Çalıştırıcı iki süreci beklemeli; RAM ve CPU paylaşılır | Hayır |
| Unsloth | HF tabanına göre yaklaşık 1,7 kat (üretici ölçümü); T4'te dolgusuz eğitim | Yeni bağımlılık; jüriye anlatılması gereken bir katman daha | **Evet** |
| ≤ 2B modellerde 4-bit yerine fp16 LoRA | %16–42 hızlı (V100 ölçümü, hakemsiz ön baskı; T4 değil) | Proje QLoRA olarak tanımlanmış (`load_in_4bit: true`) | **Evet** |
| D2 verisini küçültmek | Doğrusal | AS2'nin gücü azalır; K6 değişir | **Evet**, son çare |

## 6. "Daha çok model" için üç yol

| | A. Geniş pilot, tek ana model | B. İki model, tam deney | C. Boyut merdiveni |
| --- | --- | --- | --- |
| Ne yapılır | 6 aday mini eğitimden geçer; en iyisiyle D1 ve D2 | A'ya ek olarak ikinci bir modelle de D1 ve D2 | Qwen3 0.6B, 1.7B ve 4B ile D1 ve D2 |
| Neye cevap verir | "Hangi küçük model bu iş için en uygun?" Seçimi sağlamlaştırır | Organik verinin etkisi (AS2) modele bağlı mı? | Model boyutu ile veri karışımının etkileşimi |
| Ek GPU (tahmin) | yaklaşık +2 sa | yaklaşık +8 sa | yaklaşık +17 sa |
| Değişen kararlar | K7 (iki aday yerine altı) | K1 (tek model) ve K7. Yedek listedeki "ikinci bir küçük LLM" çekirdeğe girer | K1, K6, K7. Yeni bir araştırma sorusu doğar (K3) |
| 3 saat sınırına uyuyor mu? | D1 tek T4'te, D2 iki T4'te uyuyor | Aynı (2,4B'ye kadar) | Hayır: 4B'de D2 iki T4'le bile 4,4 sa |

Bugünkü tahminle mevcut plan (Kapı A, iki aday pilotu, Kumru ile D1 ve D2, çıkarım) yaklaşık 9 GPU saati tutuyor. İş planındaki 10 saatle uyumlu. A ile yaklaşık 11, B ile yaklaşık 19, C ile yaklaşık 26 saate çıkıyor. Haftalık kota 30 saat; A rahat, B iki haftaya yayılırsa rahat, C haftanın neredeyse tamamı.

## 7. Önerim

**Şimdi A yolunu seçelim. B'yi pilot sonucuna bakarak konuşalım.** C'yi önermiyorum: 3 saat sınırını aşıyor ve kapsamı yeni bir araştırma sorusuyla genişletiyor (K1, K3).

Pilot için önerdiğim altı aday aşağıda. Her biri bir karşılaştırma çiftinin parçası olduğu için sonuçlar yorumlanabilir olur:

| # | Aday | Bant | Mini | Neden |
| --- | --- | --- | --- | --- |
| 1 | `vngrs/Kumru-2B` | Hedef | 18 dk | Mevcut aday; sıfırdan Türkçe, en verimli tokenizer'lardan biri |
| 2 | `Qwen/Qwen3-1.7B` | Hedef | 24 dk | Mevcut aday; çok dilli |
| 3 | `newmindai/Mecellem-Qwen3-1.7B-TR` | Hedef | 24 dk | 2 numarayla aynı mimari ve tokenizer; fark yalnızca Türkçe ön eğitim |
| 4 | `utter-project/EuroLLM-1.7B-Instruct` | Hedef | 20 dk | Üçüncü bir aile; Türkçeyi resmi olarak destekliyor |
| 5 | `Qwen/Qwen3-0.6B` | Kısa | 11 dk | 2 numarayla aynı aile; boyutun ilk işareti |
| 6 | `ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1` | Kısa | 8 dk | Türkçe tokenizer'lı küçük model; 1 numarayla boyut karşılaştırması |
| (7) | `microsoft/Phi-4-mini-instruct` ya da `Qwen/Qwen3-4B-Instruct-2507` | Uzun | 34–40 dk | İsteğe bağlı üst referans, yalnızca mini eğitimle |

Altısının mini eğitimleri toplam yaklaşık 105 dakika sürüyor. Yükleme ve doğrulama setinde değerlendirmeyle birlikte bu, yaklaşık 2,5–3 GPU saati ediyor. İki T4'te ikişer ikişer koşulursa duvar saatiyle yaklaşık 1,5 saat. Altısı da gated değil, izin verici lisanslı (Apache-2.0 ya da MIT), saf transformer ve T4 riski düşük.

K6 ile ilgisi: 5 ve 6 numaralı adaylar K6'daki 1–2B aralığının altında; Kumru zaten biraz üstünde. Aralık "yaklaşık 0,5–2,5B" olarak güncellenebilir. Bu da senin kararın.

## 8. Riskler ve açık sorular

- **Süre tahmini belirsiz** (0,6–1,5 kat). Kapı B'deki kararlar tahmine değil, pilotta ölçülen hıza dayanacak.
- **Sıra değişmiyor.** Mini eğitimler de eğitimdir; Kapı A geçilmeden Kaggle'da çalıştırılmaz. Sıra yine görev 01 (veri), görev 02 (Kapı A), görev 03 (pilot).
- **fp16 kararlılığı.** Bütün adaylar bf16 ile eğitilmiş. Qwen2 ailesinde V100'de fp16 ile NaN kayıp raporları var. Qwen3 ve Llama 3.x için T4'te rapor bulunamadı; bu, sorun olmadığı anlamına gelmiyor. Pilotta her adayın kaybı izlenecek. TRL'de `fp16=True` açıkça verilmezse `bf16` varsayılan oluyor. T4'te bf16 emüle edildiği için bu yaklaşık 4 kat yavaşlık demek (ikincil).
- **DDP ile 4-bit.** Her süreç modelin tamamını kendi GPU'suna yüklemeli (`device_map={"": process_index}`). Kaggle'da 8B için DDP'de bellek taşması raporları var; 2B için sorun beklenmiyor (ikincil).
- **Cetvel skorları karşılaştırılamaz.** Cetvel'in GEC görevi GECTurk'ün sentetik testinde, sıfır-atışla ve tam eşleşmeyle ölçülüyor. Model sahiplerinin beyan ettiği skorlar (Kumru-2B 66,34, cisimcik-4b 89,55) resmi tabloyla tutmuyor; bu yüzden aday seçiminde kullanmadım. Ayrıca bu skorlar, bazı modellerin ön eğitiminde ya da SFT'sinde GECTurk benzeri veri olabileceğini düşündürüyor. Film yorumları GECTurk'ün parçası olduğu için bu bir sızıntı riski. Hazır modellerin eğitim verisini tarayamayız; raporda sınırlılık olarak yazılmalı.
- **Base modeller.** Mecellem ve Kanarya base model. İnce ayardan sonra fark etmez, ama sıfır-atış karşılaştırması anlamsız olur. Kanarya'nın şablonu da yok.
- **Literatürden aday çıkmadı.** Türker ve Eryiğit (BEA 2026) yalnızca İngilizce ve sıfır-atış çalışıyor. Güzel Turhan (2025) kelime düzeyinde yazım düzeltmeye bakıyor ve encoder ile küçük T5 modelleri kullanıyor. İlkinin bir bulgusu yine de işe yarar: "en az değişiklik" talimatı kesinliği artırıyor. Bizim talimatımız zaten bu yönde (`trgec.formats.INSTRUCTIONS`).

## 9. Son aday listesi (2026-11-23'te kesinleşti)

2026-11-21: Kullanıcı, eğitimi kolay ve Türkçeyi destekleyen diğer açık modellerin de listeye eklenmesini istedi. 7. bölümdeki altı adaya iki model eklendi: `asafaya/kanarya-2b` ve `ufakai/ufakzeka-1`.

Sıralama, ince ayardan sonra Türkçe GEC'te beklediğim başarıya göredir. **Bu bir beklentidir, ölçüm değildir;** gerçek sıralamayı pilot verecek. Dayanaklar: Türkçe ön eğitim verisinin miktarı, model boyutu ve tokenizer verimi.

| Sıra | Model | Parametre | Lisans | Kelime başına token | Mini | D1 | D2, 2×T4 | Beklentinin dayanağı |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `vngrs/Kumru-2B` | 2,38B | Apache-2.0 | 1,81 | 18 dk | 1,8 sa | 2,0 sa | Sıfırdan Türkçe, 300B token ön eğitim, instruct, verimli tokenizer |
| 2 | `newmindai/Mecellem-Qwen3-1.7B-TR` | 2,03B | Apache-2.0 | 2,74 | 24 dk | 2,2 sa | 2,6 sa | Güçlü çok dilli taban üzerine yaklaşık 225B token Türkçe ön eğitim. Base model |
| 3 | `Qwen/Qwen3-1.7B` | 2,03B | Apache-2.0 | 2,74 | 24 dk | 2,2 sa | 2,6 sa | Güçlü çok dilli taban; Türkçe resmi 119 dil arasında |
| 4 | `utter-project/EuroLLM-1.7B-Instruct` | 1,66B | Apache-2.0 | 2,37 | 20 dk | 1,9 sa | 2,2 sa | Türkçe resmi 35 dil arasında; 2024 modeli |
| 5 | `asafaya/kanarya-2b` | 2,06B | Apache-2.0 | 1,72 | 17 dk | 1,7 sa | 1,9 sa | Sıfırdan Türkçe, verimli tokenizer. Base model, şablon yok, 2024 başı; ön eğitim verisi Kumru'dan az |
| 6 | `Qwen/Qwen3-0.6B` | 0,75B | Apache-2.0 | 2,74 | 11 dk | 1,0 sa | 1,2 sa | 3 numaranın küçük kardeşi |
| 7 | `ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1` | 0,77B | MIT | 1,68 | 8 dk | 46 dk | 50 dk | Türkçe ama eski mimari (GPT-2), bağlam 1.024 |
| 8 | `ufakai/ufakzeka-1` | 0,18B | Apache-2.0 | 1,72 | 2 dk | 14 dk | 17 dk | Çok küçük; "bu iş için en az ne kadar model gerekir?" sorusunun alt sınırı |

Sekizi de gated değil, saf transformer ve T4 riski düşük. Mini eğitimlerin toplamı yaklaşık 124 dakika; yükleme ve doğrulama setinde değerlendirmeyle birlikte yaklaşık 3–3,5 GPU saati. İki T4'te ikişer ikişer koşulursa duvar saatiyle yaklaşık 1,7 saat.

Listeye girmeyenler ve nedenleri:

- `asafaya/kanarya-750m` ve Cosmos GPT-2 medium (350M): 5 ve 7 numaralarla aynı ailelerin küçükleri; ayrı bir şey ölçmezler.
- `AlicanKiraz0/Kara-Kumru-v1.0-2B`: Kumru'nun topluluk türevi.
- `meta-llama/Llama-3.2-1B-Instruct`: Türkçe resmi listede yok, gated, özel lisans.
- `boun-tabi-LMG/TURNA`: ticari olmayan lisans, decoder değil, fp16 riski.
- 3B ve üstü: 3 saat sınırını aşıyor (bölüm 3.3 ve 3.4).

Kullanıcı listeyi 2026-11-23'te onayladı. Bununla değişenler (işlendi):

- **K7:** iki aday yerine sekiz aday mini eğitimden geçer.
- **K6:** boyut aralığı "1–2B" yerine "yaklaşık 0,2–2,4B" olur.
- `configs/base.yaml` içindeki `model.candidates` sekiz kimlikle güncellenir.
- Kanarya için bir sohbet şablonu tanımlamak gerekir (görev 03, adım 1).

## 10. `max_len`: 512 yetiyor mu? (2026-11-23)

Kullanıcı sordu: token sınırı 512 olursa yeter mi, sorun çıkarır mı? Kısa cevap: **yeter.** 512, sekiz adayın hepsinde K11 kuralının (%95'lik dilim) istediğinden geniş. Asıl sorun sınırın kendisi değil, OSCAR'daki aşırı uzun satırlar; onlar kesilmemeli, elenmeli.

Ölçüm: GECTurk eğitiminden 3.000 yaklaşık örnek (bölüm 1.1'deki örnekler) ve OSCAR GEC eğitiminin ilk 20.000 satırı. Uzunluk, sohbet şablonu dahil tam eğitim örneğidir (talimat + cümle + hedef). Qwen3-1.7B, Qwen3-0.6B ve Mecellem aynı tokenizer'ı kullanıyor.

| Tokenizer | GECTurk p95 | GECTurk p99 | GECTurk en uzun | GECTurk 512 üstü | OSCAR p95 | OSCAR p99 | OSCAR 384 üstü | OSCAR 512 üstü |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kumru-2B | 213 | 275 | 555 | %0,03 | 279 | 568 | %2,4 | %1,3 |
| Qwen3 (üç model) | 286 | 358 | 696 | %0,10 | 413 | 818 | %6,0 | %3,0 |
| EuroLLM-1.7B | 283 | 356 | 660 | %0,10 | 376 | 729 | %4,7 | %2,4 |
| Kanarya-2b | 219 | 286 | 598 | %0,03 | 265 | 541 | %2,2 | %1,1 |
| Cosmos GPT-2 large | 213 | 280 | 575 | %0,03 | 257 | 522 | %2,0 | %1,1 |
| ufakzeka-1 | 207 | 270 | 552 | %0,03 | 274 | 553 | %2,4 | %1,2 |

Bulgular:

- **GECTurk'te 512 neredeyse hiçbir örneği dışarıda bırakmıyor** (%0,03–0,10).
- **OSCAR'da %1,1–3,0 örnek 512'yi aşıyor.** K11 kuralı %5'e kadar kayba izin veriyor; kurala göre sınır Kumru için 288, EuroLLM için 384, Qwen3 için 416 olurdu.
- **OSCAR'ın kuyruğu çok ağır.** 20.000 satırın en uzunu 15.475 kelime (tokenizer'a göre 48–71 bin token); binde biri 540 kelimeden uzun. Bunlar cümle değil, sayfa dökümü. Hiçbir sınır bunları kapsayamaz.
- **Cosmos GPT-2'nin bağlamı 1.024 token;** 512'lik eğitim örneği sığıyor.

Kelime sayısıyla süzme denemesi (OSCAR, 20.000 satır):

| Kaynak cümle en fazla | Tutulan satır | 512'yi aşan: Qwen3 | EuroLLM | Kumru |
| --- | --- | --- | --- | --- |
| 50 kelime | %91,2 | %0,00 | %0,01 | %0,00 |
| 70 kelime | %95,5 | %0,05 | %0,04 | %0,00 |
| 100 kelime | %97,7 | %0,77 | %0,34 | %0,04 |
| 120 kelime | %98,5 | %1,48 | %0,87 | %0,13 |

**Karar (kullanıcı 2026-11-23'te onayladı; K11 notu):**

1. **Kesme yok, eleme var.** Sınırı aşan örnek kırpılmaz, eğitimden çıkarılır. Kırpılan hedef modele yarım cevap vermeyi öğretir (K11'in gerekçesi).
2. **OSCAR alt kümesi veri hazırlığında kelime sayısıyla süzülür:** kaynak cümle en fazla 70 kelime. Süzme tokenizer'a bağlı olmadığı için sekiz aday da aynı veriyi görür. Satırların %95,5'i kalır.
3. **Pilotta bütün adaylar için tavan 512.** 70 kelime süzgecinden sonra en verimsiz tokenizer'da bile örneklerin en fazla %0,05'i tavanı aşar; onlar o model için elenir.
4. **Nihai `max_len` Kapı B'de,** seçilen modelin tokenizer'ı ve gerçek D2 alt kümesiyle K11'e göre ölçülür. `configs/base.yaml` içindeki değer o zamana kadar `null` kalır.

Sınırlılıklar ve açık noktalar:

- ~~OSCAR örneği rastgele değil~~: gerçek 12 binlik alt kümeyle yeniden ölçüldü (bölüm 11).
- ~~GECTurk hedefleri yaklaşık~~: 2026-11-24'te gerçek hedeflerle yeniden ölçüldü (bölüm 11).
- **Bellek ölçülmedi.** 512 tokenlık bir batch'in T4'e sığıp sığmadığı ve tepe bellek pilotta ölçülecek. Sığmazsa batch küçültülür, gradyan biriktirme artırılır; `max_len` küçültülmez.
- **Çıkarım ayrı bir konu.** OSCAR testinde girdilerin (yalnızca kelime sayısına bakıldı) ortalaması 25, %99'luk dilimi 122, en uzunu 453 kelime; 33 satır 100 kelimeden uzun. Tweetlerde en uzun girdi 24 kelime. Uzun satırlar eğitimde görülen uzunluğun dışında kalır ve Cosmos GPT-2'nin 1.024'lük bağlamına sığmaz. Bu satırların çıkarımda cümlelere bölünüp düzeltildikten sonra birleştirilmesi görev 02 ve 03'te tasarlanacak.

## 11. Gerçek eğitim örnekleriyle ölçüm (2026-11-24)

Bölüm 1 ve 10'daki ölçümler yaklaşık GECTurk hedefleriyle yapılmıştı. GECTurk M2 dosyaları geldi; ölçüm gerçek alt kümeler ve gerçek hedeflerle yinelendi. Komut: `python scripts/measure_lengths.py --config configs/d2_gecturk_oscar.yaml` (D1 için `configs/d1_gecturk.yaml`). Model indirilmez, yalnızca tokenizer okunur.

Örnek nasıl kuruluyor (`trgec.examples`): kayıt önce mesajlara çevrilir (GECTurk: "düzelt ve etiketle" talimatı ve `yanlış => doğru | rule_k` listesi; OSCAR: "yalnızca düzelt"), sonra modelin kendi sohbet şablonundan geçirilir ve istem ile cevap olarak ikiye ayrılır. Eğitimde kayıp yalnızca cevap kısmında hesaplanacak; çıkarımda modele yalnızca istem verilecek. İkisi aynı işlevden geldiği için eğitim ve çıkarım aynı istemi görür.

D2 (24.000 örnek: 12.000 GECTurk + 12.000 OSCAR), tam eğitim örneğinin token sayısı:

| Aday | Kelime başına token | Ortalama | %95 | %99 | En uzun | 384'ü aşan | 512'yi aşan (elenecek) | K11 ile önerilen `max_len` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kumru-2B | 1,78 | 123 | 204 | 267 | 512 | 11 | 0 | 224 |
| Qwen3-1.7B, Qwen3-0.6B, Mecellem | 2,60 | 182 | 301 | 390 | 759 | 268 | 10 | 320 |
| EuroLLM-1.7B | 2,28 | 175 | 281 | 360 | 693 | 121 | 5 | 288 |
| Kanarya-2b | 1,69 | 126 | 204 | 260 | 544 | 6 | 3 | 224 |
| Cosmos GPT-2 large | 1,64 | 120 | 196 | 250 | 515 | 4 | 1 | 224 |
| ufakzeka-1 | 1,75 | 120 | 201 | 263 | 503 | 9 | 0 | 224 |

D1'de (12.000 GECTurk örneği) sayılar benzer: %95'lik dilim 187–281, 512'yi aşan en fazla 7.

- **512 tavanı yetiyor.** 24.000 örnekten en fazla 10'u (%0,04) eleniyor; Kumru ve ufakzeka'da hiç eleme yok. Bölüm 10'daki sonuç gerçek veriyle doğrulandı.
- **Örnekler önceki tahminden kısa.** Kumru'da ortalama 123 token (bölüm 1'deki tahmin 136–140); süre tahminleri bu yüzden biraz yüksek kalıyor.
- **Türkçe tokenizer'lar belirgin biçimde verimli.** Aynı örnek Qwen3'te Kumru'ya göre yaklaşık 1,5 kat token tutuyor; bu doğrudan eğitim ve çıkarım süresine yansır.
- **K11'in yöntemi (%95'lik dilim) 224–320 öneriyor,** ama bu sınırda örneklerin %2–4'ü dışarıda kalır. Örnekler kırpılmayıp elendiği için bu, en uzun (çoğu OSCAR) örneklerin eğitimden çıkması demek. Pilotta tavan 512 kalıyor; nihai değer Kapı B'de, seçilen modelde kararlaştırılacak.
- **Kural etiketi veri setindeki haliyle yazılıyor (`rule_6`).** 10, 11, 19 ve 21 numaralı kuralların adı makalede olmadığı için ada çevrilmedi; demoda açıklama kural kartından gelir (K13).

### Bitiş işareti modelden modele değişiyor

Modelin "cevabım bitti" demek için ürettiği işaret, sohbet şablonunun cevaptan sonra yazdığı işarettir ve her zaman tokenizer'ın EOS'u değildir:

| Aday | Sohbet şablonu | Bitiş işareti | Tokenizer'ın EOS'u |
| --- | --- | --- | --- |
| Kumru-2B | var | `<\|eot_id\|>` | `<EOS>` (farklı) |
| Mecellem-Qwen3-1.7B-TR | var | `<\|im_end\|>` | `<\|endoftext\|>` (farklı) |
| Qwen3-1.7B, Qwen3-0.6B | var | `<\|im_end\|>` | aynı |
| EuroLLM-1.7B | var | `<\|im_end\|>` | aynı |
| Kanarya-2b | **yok** | `</s>` | aynı |
| Cosmos GPT-2 large | var, ama cevaptan sonra işaret yazmıyor | `<\|stop\|>` (biz ekliyoruz) | aynı |
| ufakzeka-1 | var | `<\|im_end\|>` | `<\|endoftext\|>` (farklı) |

- **Neden önemli:** eğitimde cevabın sonuna yanlış işaret konursa ya da çıkarımda yanlış işarette durulursa model cümleyi bitirdikten sonra yazmaya devam eder; bozuk çıktı oranı yükselir ve süre uzar. Üç adayda (Kumru, Mecellem, ufakzeka) yalnızca tokenizer'ın EOS'una güvenmek bu hataya yol açardı.
- **Çözüm:** `trgec.examples.render` cevabı şablonun yazdığı işaretle birlikte alır; `stop_text` aynı işareti çıkarım için verir. Sekiz adayın hepsinde işaret tek token.
- **Kanarya'nın şablonu yok;** sabit bir yedek şablon kullanılıyor (`Kullanıcı: ...` ve `Asistan:`). Cosmos GPT-2'nin şablonu cevaptan sonra yalnızca satır sonu yazdığı için EOS'u biz ekliyoruz.
- **Qwen3'te düşünme modu kapalı;** istem boş bir `<think>` bloğuyla bitiyor ve cevap doğrudan başlıyor.

## Kaynaklar

**Doğrulandı (HF API ve model kartı, 2026-11-21):** Bölüm 2 ve 3'teki bütün kimlikler, parametre sayıları, lisanslar ve gated durumu; EuroLLM-1.7B-Instruct ve Phi-4-mini-instruct kartlarında `tr` dil etiketi.

**İkincil (alt ajan araştırmaları, 2026-11-20/21):**

- T4 hız ölçümleri: Unsloth not defterlerinin kayıtlı çıktıları (`github.com/unslothai/notebooks/nb`: Qwen2.5-7B, Llama-3.1-8B, Gemma2-2B, Llama-3.2-3B, Qwen3-4B, Phi-3.5); Unsloth README'sindeki HF karşılaştırması (commit `a1da50b`); implicit-none.com (Qwen2.5-0.5B, HF LoRA); github.com/Azoqoz/LoRA-Fine-Tuning-Pipeline (Qwen2.5-1.5B, HF QLoRA).
- fp16 ve LoRA hızı: arXiv 2608.23816 (V100, hakemsiz); TRL `SFTConfig` belgesi; earino.github.io (T4'te bf16 ölçümü).
- Kaggle sınırları: gabrubuilds.com ve nandigamharikrishna.substack.com. İkisi de Kaggle belgelerini alıntılıyor; resmi sayfa JavaScript ile yüklendiği için okunamadı.
- Qwen3 dil listesi: qwenlm.github.io/blog/qwen3. Qwen3.5 ve Triton: transformers `qwen3_5` belgesi, unsloth.ai/docs/models/qwen3.5, github.com/triton-lang/triton.
- Cetvel: Er vd., EACL 2026 (aclanthology.org/2026.eacl-long.46); görev tanımı github.com/KUIS-AI/cetvel.
- Mecellem: arXiv 2601.16018. ufakzeka: arXiv 2609.25081.
- Türker ve Eryiğit 2026 (aclanthology.org/2026.bea-1.17); Güzel Turhan 2025 (PMC12192738).
