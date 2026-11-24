# Pilot raporu: sekiz adayın mini eğitimi

Görev 03, adım 6 ve 7. Tarih: 2026-11-24. Bu rapor Kapı B'de (ana model seçimi) kullanılacak ölçümleri toplar. Seçimi kullanıcı yapar.

## 1. Ne yapıldı

- **Ortam:** Kaggle, tek T4 (15 GB). Sekiz aday tek koşuda sırayla çalıştı; koşu 2 saat 55 dakika sürdü. Mecellem, yükleme hatası düzeltildikten sonra tek başına yeniden koşturuldu (bölüm 3.1).
- **Eğitim:** her aday aynı 2.000 örnekle (1.000 GECTurk + 1.000 OSCAR, tohum 42), 1 tur, 125 adım. Ayarlar `configs/base.yaml` içindeki gibi: 4-bit NF4, LoRA (r 16, alpha 32, dropout 0,05, bütün doğrusal katmanlar), fp16, öğrenme oranı 2e-4, etkin grup 16.
- **Ölçüm:** her aday aynı üç doğrulama setinde tahmin üretti: `gecturk_dev_2k` (2.000 cümle: 1.000 hatalı, 1.000 hatasız), `oscar_dev` (500 cümle), `oscar_dev_clean` (250 hatasız cümle). Test setlerine dokunulmadı.
- **Skorlama:** tahminler indirildi ve yerelde ERRANT-TR ile skorlandı (`scripts/summarize_pilot.py`). Ana mod `cs` (K15).
- **Sürümler (Kaggle'da kurulan):** torch 2.11.0+cu130, transformers 5.24.0, peft 0.22.0, accelerate 1.16.0, bitsandbytes 0.50.2. `requirements-kaggle.txt` bu sürümlere sabitlendi.

Pilotun sınırı: tek tohum, 2.000 örnek. Buradaki sıralama 24.000 örnekli ana eğitimde değişebilir; pilot yalnızca adayları kabaca ayırır (K7).

## 2. Eğitim ölçümleri

| Aday | Parametre (4-bit sayım) | Grup x biriktirme | sn/adım | Eğitim (sn) | Tepe bellek (GB) | Kayıp ilk → son | Durum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `vngrs/Kumru-2B` | 1,36B | 4 x 4 | 2,317 | 400 | 11,86 | 0,58 → 0,20 | tamam |
| `newmindai/Mecellem-Qwen3-1.7B-TR` | 1,03B | 2 x 8 | 3,346 | 539 | 8,63 | 0,53 → 0,32 | tamam (ikinci koşu; ilk koşuda öğrenmedi, bölüm 3.1) |
| `Qwen/Qwen3-1.7B` | 1,03B | 2 x 8 | 3,436 | 529 | 8,63 | 0,51 → 0,15 | tamam |
| `utter-project/EuroLLM-1.7B-Instruct` | 1,10B | 4 x 4 | 2,263 | 365 | 11,85 | 0,42 → 0,17 | tamam |
| `asafaya/kanarya-2b` | 1,13B | 4 x 4 | | 154 | 14,29 | NaN | başarısız (bölüm 3.2) |
| `Qwen/Qwen3-0.6B` | 0,39B | 4 x 4 | 1,717 | 278 | 9,82 | 0,51 → 0,21 | tamam |
| `ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1` | 0,43B | 4 x 4 | 1,259 | 225 | 10,12 | 0,85 → 0,30 | tamam |
| `ufakai/ufakzeka-1` | 0,11B | 8 x 2 | 0,664 | 126 | 7,27 | 1,53 → 0,41 | tamam |

Notlar:

- "Parametre" sütunu 4-bit yüklenmiş modelde sayılan değerdir; nicemlenmiş katmanlar daha az sayıldığı için modelin adındaki sayıdan küçüktür. "Eğitim (sn)" bellek taşmasıyla biten denemeleri de içerir.
- **Grup boyutu 16 hiçbir adayda belleğe sığmadı.** Yarıya indirme mekanizması (`trgec.pilot`) hepsinde devreye girdi: ufakzeka 8'de, Qwen3-1.7B ve Mecellem 2'de, diğerleri 4'te çalıştı. Etkin grup hep 16 kaldı.
- Bellek ihtiyacı beklenenden yüksek: 2B'lik adaylar grup 4'te 12 GB'a yaklaşıyor. `training.gradient_checkpointing` kapalı; açmak belleği düşürür, hızı azaltır. Pilotta denenmedi.
- Sağlıklı yedi adayın hiçbirinde NaN görülmedi.

## 3. Sorunlu iki aday

### 3.1 Mecellem: çıktı katmanı boş yükleniyor

- **Belirti:** kayıp ilk adımdan son adıma 11,93, gradyan normu 0. 11,93 = ln(151.936), yani model sözlükteki her tokena eşit olasılık veriyor. Çıkarımda bütün çıktılar `!!!!...` (0 numaralı token).
- **Neden (doğrulandı):** modelin ağırlık dosyasında (`model.safetensors`, depo sürümü `107c316`) `lm_head.weight` ayrıca kayıtlı ve örneklenen bütün değerleri sıfır; `model.embed_tokens.weight` dolu. Yapılandırma `tie_word_embeddings: true` diyor, ama transformers 5.24 iki tabloyu farklı görünce bağlamıyor ve bunu günlükte uyarı olarak yazıyor. Dosyanın tamamı indirilmeden, ilgili tensörlerin dört ayrı bölgesinden 8.192 değer okunarak bakıldı.
- **Düzeltme:** `trgec.modeling.restore_tied_output_layer`. Yalnızca üç koşul birlikte sağlanırsa çalışır: yapılandırma bağlama istiyor, iki tablo ayrı, çıktı tablosunun tamamı sıfır. Bu durumda çıktı katmanı gömme katmanına bağlanır (modelin yapılandırmasının zaten istediği şey). Sağlam yüklenen modellere dokunmaz. `train.py` ve `predict.py` yüklemeden hemen sonra çağırır.
- **Yeniden koşu (aynı gün):** onarım çalıştı. Kayıp 0,53 → 0,32, gradyan normu sıfır değil, tepe bellek Qwen3-1.7B ile aynı (8,63 GB; ilk koşudaki fazlalık, bağlanmamış ikinci tablodan geliyordu).

**İkinci sorun: model cevabı bitirip duramıyor.** Yeniden koşuda Mecellem düzeltilmiş cümleyi ve listeyi doğru yazdı, ama bitiş işaretini (`<|im_end|>`) üretemedi; onun yerine rastgele görünen bir token yazıp aynı bloğu sınıra kadar yineledi. 2.750 çıktının tamamı çıktı sınırına takıldı, %97'si bozuk biçimde sayıldı ve çıkarım 41 dakika sürdü.

- **Neden (doğrulandı):** Mecellem bir "base" model; sürekli ön eğitimi düz metinle yapılmış. Gömme tablosunda `<|im_start|>` ve `<|im_end|>` satırları birebir aynı (kosinüs 1,00) ve eğitilmemiş başka tokenlarla neredeyse aynı (çıktıda görülen `ﳈ` ile kosinüs 0,97). Çıktı katmanı gömme tablosuna bağlı ve LoRA bu tabloyu eğitmiyor; model bu satırlar arasında ayrım yapamıyor. Qwen3-1.7B'de aynı iki satır farklı (kosinüs 0,67). `<|endoftext|>` satırı Mecellem'de de sağlam (diğerleriyle kosinüs 0,13, normu sıradan tokenlar gibi).
- **Düzeltme:** `model.plain_prompt` listesindeki adaylarda sohbet şablonu kapatılır (`trgec.examples.use_plain_prompt`): istem düz metin olur (`Kullanıcı: ...` / `Asistan:`; Kanarya için zaten kullanılan yedek şablon) ve cevap `<|endoftext|>` ile biter. Yerelde Mecellem'in tokenizer'ıyla doğrulandı. **Bu düzeltmeyle Kaggle koşusu henüz yapılmadı.**
- **Aşağıdaki Mecellem skorları duramayan çıktılarla ölçüldü.** Düzeltilmiş cümle çıktının ilk `###` işaretinden önceki kısmından alındığı için ERRANT-TR skoru anlamlıdır; bozuk çıktı oranı ve hız ise bu soruna aittir, modelin kalıcı özelliği değildir.

### 3.2 Kanarya: fp16'da kayıp NaN

- **Belirti:** grup 4 x 4 ile eğitim başladı, 20. adımdaki ilk kayıt NaN. İzleyici eğitimi durdurdu (56 sn). Tepe bellek 14,29 GB, yani T4'ün sınırında.
- **Olası neden:** fp16'da sayısal taşma. Kanarya GPT-J mimarisinde ve bf16 ile eğitilmiş; T4 bf16'yı desteklemiyor. Bu açıklama ayrıca sınanmadı.
- **Seçenekler:** (a) Kanarya'yı listeden çıkarmak. K12 "fp16 kararsızlığı riski olan modellerden kaçınılır" diyor. (b) Yalnızca bu adayı fp32 hesapla yeniden denemek. Bellek zaten sınırda olduğu için grup 1 ya da 2'ye düşer, eğitim birkaç kat yavaşlar ve aday diğerleriyle aynı ayarda ölçülmemiş olur.
- **Durum:** karar kullanıcıda; kendiliğimden yeniden koşturmadım.

## 4. Skorlar (ERRANT-TR, `cs`)

Güven aralıkları cümle düzeyinde bootstrap ile hesaplandı (%95).

### 4.1 `gecturk_dev_2k`: sentetik, kurala dayalı hatalar

| Aday | P | R | F0.5 | %95 güven aralığı | Bozuk çıktı (%) | Hatasız cümlede aşırı düzeltme (%) | Cümle/sn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `newmindai/Mecellem-Qwen3-1.7B-TR` (duramayan çıktılar) | 87,8 | 60,2 | 80,4 | 78,4–82,2 | 97,7 | 3,3 | 1,23 |
| `vngrs/Kumru-2B` | 81,5 | 50,9 | 72,7 | 70,5–75,0 | 0,7 | 4,0 | 6,20 |
| `ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1` | 69,8 | 54,8 | 66,2 | 63,6–68,5 | 20,7 | 11,7 | 4,89 |
| `utter-project/EuroLLM-1.7B-Instruct` | 80,0 | 36,5 | 64,6 | 62,0–67,6 | 0,2 | 2,1 | 4,06 |
| `Qwen/Qwen3-1.7B` | 77,9 | 32,9 | 61,2 | 58,3–64,0 | 0,0 | 3,7 | 2,73 |
| `Qwen/Qwen3-0.6B` | 68,3 | 23,4 | 49,4 | 46,4–52,5 | 0,2 | 4,7 | 3,21 |
| `ufakai/ufakzeka-1` | 18,1 | 28,6 | 19,5 | 17,8–21,2 | 7,5 | 45,0 | 4,78 |

### 4.2 `oscar_dev`: organik yazım hataları

| Aday | P | R | F0.5 | %95 güven aralığı | Bozuk çıktı (%) | Cümle/sn |
| --- | --- | --- | --- | --- | --- | --- |
| `vngrs/Kumru-2B` | 76,3 | 61,8 | 72,9 | 70,5–75,2 | 9,6 | 3,12 |
| `newmindai/Mecellem-Qwen3-1.7B-TR` (duramayan çıktılar) | 72,2 | 69,9 | 71,7 | 69,4–73,7 | 95,4 | 0,91 |
| `ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1` | 69,8 | 70,2 | 69,9 | 66,7–72,6 | 22,8 | 2,70 |
| `utter-project/EuroLLM-1.7B-Instruct` | 72,9 | 49,1 | 66,5 | 64,1–69,0 | 3,8 | 2,71 |
| `Qwen/Qwen3-1.7B` | 68,9 | 43,0 | 61,5 | 58,9–64,0 | 3,4 | 1,22 |
| `Qwen/Qwen3-0.6B` | 62,9 | 23,2 | 46,9 | 44,2–49,7 | 1,6 | 1,62 |
| `ufakai/ufakzeka-1` | 38,7 | 41,4 | 39,2 | 36,4–42,0 | 32,6 | 2,00 |

### 4.3 `oscar_dev_clean`: hatasız OSCAR cümleleri

Bu sette düzeltilecek bir şey yok; ölçülen, modelin hatasız cümleyi değiştirme oranıdır.

| Aday | Aşırı düzeltme (%) | Bozuk çıktı (%) | Cümle/sn |
| --- | --- | --- | --- |
| `Qwen/Qwen3-0.6B` | 15,2 | 1,2 | 1,98 |
| `utter-project/EuroLLM-1.7B-Instruct` | 20,8 | 1,6 | 2,75 |
| `Qwen/Qwen3-1.7B` | 22,0 | 0,8 | 1,83 |
| `vngrs/Kumru-2B` | 33,2 | 8,4 | 3,86 |
| `ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1` | 47,2 | 26,4 | 3,39 |
| `newmindai/Mecellem-Qwen3-1.7B-TR` (duramayan çıktılar) | 48,8 | 99,2 | 0,93 |
| `ufakai/ufakzeka-1` | 92,8 | 27,6 | 2,23 |

Bu sayı dikkatle okunmalı: OSCAR hedefleri yalnızca kısmen düzeltilmiş (`03_TEKNIK_NOTLAR.md`), yani "hatasız" sayılan bir cümlede gerçek bir hata kalmış olabilir ve modelin onu düzeltmesi burada aşırı düzeltme sayılır. Kullanıcının elle kontrolü bu yüzden bekliyor.

### 4.4 Skorlardan çıkanlar

- **İki aday öne çıkıyor: Mecellem ve Kumru-2B.** GECTurk'te Mecellem açık farkla birinci; güven aralığı Kumru'nunkiyle örtüşmüyor (78,4–82,2 ve 70,5–75,0). OSCAR'da ikisi başa baş (72,9 ve 71,7; aralıklar örtüşüyor). Kumru'nun aralığı GECTurk'te üçüncü adayla örtüşmüyor (63,6–68,5).
- **Mecellem ile Qwen3-1.7B aynı mimari ve aynı tokenizer;** tek fark Mecellem'in Türkçe sürekli ön eğitimi. Bu fark GECTurk'te 19, OSCAR'da 10 puan F0.5 ediyor. İstem biçimleri henüz aynı değil (Mecellem düzeltmeden sonra düz istem kullanacak), o yüzden bu karşılaştırma kesin değil.
- **Cosmos GPT-2** F0.5'te üçüncü, ama çıktılarının beşte biri bozuk biçimde ve GECTurk'ün hatasız cümlelerini ufakzeka'dan sonra en çok o değiştiriyor.
- **EuroLLM ve Qwen3-1.7B** temkinli: kesinlik yüksek, duyarlılık düşük, biçim neredeyse kusursuz.
- **Model küçüldükçe** (AS1): Qwen3-1.7B'den 0.6B'ye inince F0.5 GECTurk'te 11,8, OSCAR'da 14,6 puan düşüyor; OSCAR'da kaybın çoğu duyarlılıkta (43,0 → 23,2). ufakzeka (0,1B) 2.000 örnekle kullanılabilir bir düzeye gelmiyor.
- ufakzeka'nın skorları deneme koşusuyla aynı (19,5 ve 39,2): cümleye bağlı çıktı sınırı (`output_budget`) sonucu değiştirmedi, çıkarımı 954 saniyeden 809 saniyeye indirdi.

## 5. Düzeltme listesi ve kural etiketi (iç analiz)

Bu bölümdeki sayılar yayımlanmış bir skorlayıcıdan değil, kendi sayımımızdan gelir; yalnızca iç kullanım içindir. Soru: model cümleyi düzeltirken `###` altına "yanlış => doğru | kural" satırını da yazıyor mu, yazdığı kural doğru mu? Ölçüm `gecturk_dev_2k` setinin hatalı 1.000 cümlesinde yapıldı.

| Aday | Değiştirdiği hatalı cümle | Bunların kaçında liste yazdı (%) | Listedeki düzeltmelerin altınla eşleşmesi: P / R (%) | Eşleşenlerde doğru kural (%) |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3-1.7B` | 491 | 95,5 | 67,8 / 23,3 | 47,3 |
| `Qwen/Qwen3-0.6B` | 407 | 88,2 | 57,9 / 13,7 | 59,5 |
| `newmindai/Mecellem-Qwen3-1.7B-TR` (duramayan çıktılar) | 755 | 79,5 | 18,8 / 33,2 | 41,8 |
| `ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1` | 760 | 40,8 | 12,3 / 3,0 | 38,8 |
| `utter-project/EuroLLM-1.7B-Instruct` | 536 | 28,4 | 28,8 / 3,9 | 41,5 |
| `ufakai/ufakzeka-1` | 859 | 4,7 | 10,5 / 0,4 | 50,0 |
| `vngrs/Kumru-2B` | 678 | 4,0 | 37,5 / 0,9 | 20,0 |

Sayım yalnızca çıktının ilk cevap bloğuna (ilk `###` ve ardından gelen satırlar, ikinci `###` işaretine kadar) bakar; böylece Mecellem'in yinelenen blokları sayılmaz.

- **Kumru cümleyi düzeltiyor ama listeyi yazmıyor:** değiştirdiği 678 hatalı cümlenin 651'inde `###` altına `YOK` yazdı. ERRANT-TR yalnızca düzeltilmiş cümleye baktığı için bu, bölüm 4'teki skorlarda görünmüyor.
- Qwen3 ailesi (Mecellem dahil) listeyi çoğu zaman yazıyor; yine de eşleşen düzeltmelerin yarısında kural yanlış. Mecellem'in listesinde gereksiz satır çok (kesinlik %18,8): değişmeyen kelimeleri de `sitelere => sitelere` gibi listeye yazıyor.
- **Yorum sınırlı:** pilot verisinin yalnızca dörtte biri (yaklaşık 500 örnek) dolu bir liste içeriyor; 25 kural için bu çok az. D2 ana eğitiminde bu sayı 6.000. Kumru'nun listeyi daha çok veriyle öğrenip öğrenmediği pilottan bilinemez. Projenin amacı ihlal edilen kuralı da göstermek olduğu için bu, Kapı B'den önce bilinmesi gereken en önemli açık soru.

## 6. Süre tahmini (adım 7)

Ölçülen adım süresinden: D1 12.000 örnek = 750 adım, D2 24.000 örnek = 1.500 adım. Test çıkarımı, doğrulama setlerinde ölçülen hızdan üç test seti (4.708 cümle) için hesaplandı.

| Aday | D1 eğitimi | D2 eğitimi | Üç test setinde çıkarım |
| --- | --- | --- | --- |
| `vngrs/Kumru-2B` | 29 dk | 58 dk | 26 dk |
| `Qwen/Qwen3-1.7B` | 43 dk | 86 dk | 64 dk |
| `newmindai/Mecellem-Qwen3-1.7B-TR` | 42 dk | 84 dk | ölçülemedi (çıktılar durmadı) |
| `utter-project/EuroLLM-1.7B-Instruct` | 28 dk | 57 dk | 30 dk |
| `Qwen/Qwen3-0.6B` | 21 dk | 43 dk | 48 dk |
| `ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1` | 16 dk | 31 dk | 30 dk |
| `ufakai/ufakzeka-1` | 8 dk | 17 dk | 39 dk |

- **D2 her adayda tek oturuma (12 saat) rahat sığıyor;** veri boyutunu düşürmek gerekmiyor. Tahmin kabadır: adım süresi gruptaki cümle uzunluklarına bağlı ve D1'de OSCAR yok.
- Kaggle'ın haftalık 30 saatlik GPU kotasından bugüne kadar yaklaşık 4,5 saat kullanıldı.

## 7. B planı ölçütleri (K21)

| Ölçüt | Durum |
| --- | --- |
| (a) D2 tek oturuma sığmıyor ve veri küçültülünce skor çöküyor | Geçerli değil: D2 en yavaş adayda yaklaşık 1,5 saat. |
| (b) fp16 kararsızlığı iki adayda da çözülemiyor | Geçerli değil: yedi adayda NaN yok; yalnızca Kanarya'da görüldü. |
| (c) Mini eğitimli model doğrulamada eğitimsiz halinden iyi değil | **Doğrudan ölçülmedi.** Pilotta eğitimsiz (sıfır-atış) çıkarım yapılmadı. Dolaylı gösterge: cümleyi hiç değiştirmeyen bir sistemin F0.5'i 0'dır, mini eğitimli Mecellem 80,4, Kumru 72,7 alıyor. Kesin kontrol için seçilen adayın eğitimsiz hali aynı setlerde bir kez ölçülmeli (yaklaşık 10 dakika GPU). |

## 8. Kapı B için değerlendirme

Seçim kullanıcının. Ölçümlerin gösterdiği:

- **Mecellem-Qwen3-1.7B-TR** GECTurk doğrulamasında en yüksek skoru alıyor, OSCAR'da Kumru ile başa baş ve listeyi çoğu zaman yazıyor. Açık yanları: (1) durma sorunu için yazılan düzeltme henüz Kaggle'da sınanmadı; bozuk çıktı oranı, hız ve hatasız OSCAR cümlelerindeki %48,8'lik aşırı düzeltme bu sorunla birlikte ölçüldü ve yeniden ölçülmeli; (2) en yavaş eğitilen adaylardan (D2 yaklaşık 84 dakika; yine de oturuma rahat sığıyor); (3) listesinde gereksiz satır çok.
- **Kumru-2B** OSCAR'da birinci, GECTurk'te ikinci, en hızlı çıkarımı yapıyor ve biçimi GECTurk'te temiz. Türkçe için sıfırdan eğitilmiş ve tokenizer'ı verimli (rapor `03_model_adaylari.md`). Zayıf yanları: pilotta düzeltme listesini yazmıyor, OSCAR tarzı metinde çıktılarının yaklaşık onda biri bozuk biçimde, hatasız OSCAR cümlelerinin üçte birini değiştiriyor.
- **Kumru için bir çekince:** `03_model_adaylari.md` (bölüm 8), Kumru'nun ön eğitiminde ya da talimat eğitiminde GECTurk benzeri veri bulunmuş olabileceğini not ediyor. Bunu sınayamıyoruz.
- **Qwen3-1.7B** çıktı biçimini ve listeyi en temiz öğrenen aday; düzeltme başarısı Kumru'dan 11 puan geride.
- **EuroLLM-1.7B** Kumru'dan 7–8 puan geride, biçimi temiz, listeyi çoğu zaman yazmıyor.
- **Cosmos GPT-2** küçük boyutuna göre güçlü, ama bozuk çıktı oranı yüksek.

Önerim: Kapı B'yi şimdi kapatmamak. Seçim Mecellem ile Kumru arasında görünüyor ve ikisinin de birer açık sorusu var; Kapı B'den sonra model değişmiyor. Kapatmadan önce tek bir Kaggle koşusunda şunlar ölçülmeli:

1. Mecellem, düz istem düzeltmesiyle aynı 2.000 örnekte yeniden (duruyor mu, bozuk çıktı ve aşırı düzeltme kaça iniyor?).
2. Mecellem ve Kumru, daha büyük bir pilotla (örneğin 3.000 GECTurk + 3.000 OSCAR): Kumru listeyi yazmayı öğreniyor mu, sıralama korunuyor mu?
3. Seçilecek adayın eğitimsiz hali aynı setlerde (K21 c).

Tahmini süre: toplam bir saat kadar GPU. Hepsi doğrulama setlerinde; test setlerine dokunulmaz.

Kapı B'de ayrıca doldurulacaklar: `training.max_len` (ölçüme göre 512), örnekleme boyutları (12.000 + 12.000 değişmeden kalabilir), `model.chosen`.

## 9. Dosyalar

- Ham çıktılar: `outputs/kaggle/pilot/` ve `outputs/kaggle/pilot_mecellem/`; ikisinin birleşimi `outputs/kaggle/pilot_birlesik/` (depoda değil). Tahminler: `outputs/predictions/pilot__<aday>/`. Skorlar: `outputs/eval/pilot__<aday>/`.
- Tablo: `outputs/pilot/pilot_table.md` ve `.json` (`scripts/summarize_pilot.py` üretir).
