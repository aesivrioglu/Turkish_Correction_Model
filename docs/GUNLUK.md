# Günlük

Her çalışma oturumunun sonunda en üste yeni bir kayıt eklenir. Amaç, bir sonraki oturumun (ve Claude Code'un) kaldığı yerden devam edebilmesidir.

Kayıt biçimi:

```
## YYYY-AA-GG, görev NN
- Yapılan:
- Ölçülen (sayılarla):
- Karar ya da değişiklik (varsa K numarası):
- Sorun ya da açık soru:
- Sıradaki adım:
```

## 2026-11-24, görev 03 (adım 6 hazırlığı: Kaggle pilotu)

- Yapılan:
  - Kullanıcı onayıyla `gorev-01-03-veri-kapi-a-egitim` dalı `main`'e birleştirildi (ileri sarma).
  - `gecturk_dev_2k` üretildi: GECTurk dev'den katmanlı 2.000 cümle (1.000 hatalı + 1.000 hatasız, kural başına en az 20). `prepare_data.py` ve sızıntı taramasına eklendi; `evaluate.py` bu seti ERRANT-TR ile skorluyor.
  - `scripts/run_pilot.py` yazıldı: adayları sırayla ayrı süreçlerde eğitir, doğrulama setlerinde tahmin üretir, hata veren adayı atlayıp devam eder, sonuçları `pilot_outputs.zip` içine koyar. Tek GPU görünür kılınıyor (`CUDA_VISIBLE_DEVICES=0`).
  - `kaggle/calistirici.ipynb` pilot için yeniden yazıldı: depo klonu, kurulum, veriyi Kaggle veri setinden kopyalama, `run_pilot.py`, zip. `HF_TOKEN` artık şart değil.
  - Kaggle'a yüklenecek paket hazırlandı: `outputs/kaggle_dataset/trgec-pilot-veri.zip` (2,4 MB; yalnızca eğitim ve doğrulama dosyaları).
- Ölçülen:
  - `run_pilot.py` yerelde duman modunda çalıştı (ufakzeka-1: eğitim 43 sn, üç sette tahmin 3/3).
  - `gecturk_dev_2k` ile `gecturk_train_12k`, film yorumları, OSCAR testi ve tweetler arasında çakışma 0.
  - GitHub deposu herkese açık (anonim istek 200 döndü); Kaggle klonu için anahtar gerekmiyor.
  - Testler: 93 geçti.
- Karar ya da değişiklik:
  - Pilotta GECTurk doğrulaması 2.000 cümlelik sabit alt kümeyle yapılır (tamamı 20.768 cümle; sekiz adayda çıkarım çok uzun sürer). `sampling.gecturk_dev_n: 2000`.
  - Pilot doğrulama skorları ERRANT-TR ile hesaplanır (yayımlanmış bir skorla kıyas yok; K15 test setleri içindir).
  - Veri Kaggle'a özel Kaggle veri seti olarak taşınır; test setleri pakete konmaz.
- Sorun ya da açık soru:
  - 4-bit ve fp16 yolu ilk kez Kaggle'da çalışacak; önce tek adayla (ufakzeka-1) deneme koşusu yapılacak.
  - Kaggle'daki `bitsandbytes` sürümü sabitlenmedi; deneme koşusunda çalışan sürüm yazılacak.
  - Kullanıcının elle kontrolü ve M2 scorer hâlâ bekliyor.
- Sıradaki adım: Veri setini Kaggle'a yükle, not defterini çalıştır (deneme koşusu), `pilot_outputs.zip` dosyasını indir; yerelde skorla. Deneme temizse `MODELS = ""` ile sekiz adayın pilotu.

## 2026-11-24, görev 03 (adım 4 ve 5: `train.py`, `predict.py`, duman testi)

- Yapılan:
  - Kullanıcı `peft` ve `accelerate` bağımlılıklarını onayladı; kuruldu (`peft` 0.22.0, `accelerate` 1.16.0). `requirements.txt` güncellendi; `requirements-kaggle.txt` içinde `transformers`, `peft` ve `accelerate` yerelde sınanan sürümlere sabitlendi.
  - `trgec.examples`: `encode` (istem konumları kayıptan çıkarılır, yalnızca cevap tokenları kayba girer) ve `pad_batch`. 2 yeni test.
  - `scripts/train.py` yazıldı: karışımı okur, örnekleri kurar, tavanı aşanı eler, modeli yükler (Kaggle'da 4-bit NF4 + fp16; duman testinde fp32 CPU), LoRA ekler, `transformers.Trainer` ile eğitir (uzunluğa göre gruplama, NaN kayıpta durma, kayıt noktası ve `--resume`), adaptörü ve `run_meta.json` özetini yazar.
  - `scripts/predict.py` yazıldı: temel model + adaptör, eğitimdekiyle aynı istem, açgözlü üretim, bitiş işaretinde durma, sol dolgu, `parse_output` ile ayrıştırma, isteğe bağlı emniyet kemeri. Çıktı doğrudan `evaluate.py`'ye verilebilir.
  - `configs/pilot.yaml` (her kaynaktan 1.000 örnek) eklendi; `base.yaml` içine eğitim ve çıkarım ayarları eklendi.
- Ölçülen (yalnızca kodun çalıştığını gösterir, başarı ölçüsü değildir):
  - Duman testi (`ufakai/ufakzeka-1`, 8 örnek, CPU): 2 adım hatasız; adaptör kaydedildi. Öğrenen parametre 5,2 milyon / 187,7 milyon (%2,78).
  - Aynı 8 örnekte 30 adım: kayıp 3,19'dan 0,17'ye düştü (maske, gradyan ve adaptörler çalışıyor).
  - Bu adaptörle `predict.py`: 8 cümlenin 4'ünde çıktı hedefle birebir aynı, 6'sında biçim düzgün, hiçbiri çıktı sınırına takılmadı (üretim bitiş işaretinde duruyor).
  - Testler: 92 geçti.
- Karar ya da değişiklik: `base.yaml` içine pilot için başlangıç ayarları eklendi: `batch_size` 16, `grad_accum` 1, `warmup_ratio` 0,03, `max_grad_norm` 1,0, `lora_target_modules` `all-linear`, `save_steps` 200, `inference.max_new_tokens` 256. Bunlar pilotta ölçülüp gerekirse değişir; Kapı B'de dondurulur.
- Sorun ya da açık soru:
  - **Veri Kaggle'a nasıl gidecek?** `data/processed` depoya girmiyor; `docs/04_ORTAM_VE_IS_AKISI.md` bunu söylemiyor. Kullanıcı kararı gerekiyor (özel Kaggle veri seti öneriliyor).
  - Kaggle not defteri hâlâ yalnızca `train.py` çağırıyor; pilot için `--model` ve `predict.py` adımları eklenecek.
  - 4-bit yükleme ve fp16 yolu yerelde sınanamıyor (GPU ister); ilk Kaggle koşusunda tek adayla denenmeli.
  - Uzun test satırlarının çıkarımda cümlelere bölünmesi henüz yok.
  - Kullanıcının elle kontrolü, M2 scorer ve commit hâlâ bekliyor.
- Sıradaki adım: Verinin Kaggle'a taşınma yolu kararlaştırılınca not defterini pilot için güncelle; tek adayla (en küçüğü) Kaggle'da 4-bit yolunu dene; ardından sekiz adayın mini eğitimleri.

## 2026-11-24, görev 03 (adım 2 ve 3: eğitim örnekleri ve token uzunluğu)

- Yapılan:
  - `src/trgec/examples.py`: kayıttan mesajlara (`training_messages`, `inference_messages`), mesajlardan modelin sohbet şablonuyla istem ve cevaba (`render`), bitiş işareti (`stop_text`), D1/D2/mini karışımı (`mix_records`), tavanı aşanları eleme (`drop_over_length`). 11 test.
  - `scripts/measure_lengths.py`: sekiz adayın tokenizer'ıyla gerçek eğitim örneklerinin uzunluğunu ölçer; sonuç `outputs/lengths/` altında.
  - `raporlar/03_model_adaylari.md` içine bölüm 11 eklendi.
- Ölçülen (D2, 24.000 örnek):
  - 512'yi aşan örnek: Kumru 0, ufakzeka 0, Cosmos GPT-2 1, Kanarya 3, EuroLLM 5, Qwen3 ailesi 10.
  - Ortalama / %95: Kumru 123 / 204; Qwen3 182 / 301; EuroLLM 175 / 281; Kanarya 126 / 204; Cosmos 120 / 196; ufakzeka 120 / 201.
  - Kelime başına token: Cosmos 1,64; Kanarya 1,69; ufakzeka 1,75; Kumru 1,78; EuroLLM 2,28; Qwen3 2,60.
  - K11 yöntemiyle önerilen `max_len`: 224 (Kumru, Kanarya, Cosmos, ufakzeka), 288 (EuroLLM), 320 (Qwen3); bu sınırlarda örneklerin %2–4'ü elenir.
  - Testler: 90 geçti.
- Karar ya da değişiklik:
  - Kural etiketi model çıktısında veri setindeki haliyle (`rule_6`) yazılıyor.
  - Kanarya için yedek sohbet şablonu tanımlandı; Cosmos GPT-2'de cevabın sonuna EOS ekleniyor.
- Sorun ya da açık soru:
  - Bitiş işareti üç adayda (Kumru, Mecellem, ufakzeka) tokenizer'ın EOS'undan farklı; eğitim ve çıkarım `stop_text` üzerinden yürümeli.
  - `train.py` için yeni bağımlılıklar gerekiyor (`peft`, `accelerate`; Kaggle'da ayrıca `bitsandbytes`); kullanıcı onayı bekleniyor.
  - Kullanıcının elle kontrolü, M2 scorer ve commit hâlâ bekliyor.
- Sıradaki adım: Bağımlılık onayı gelince `scripts/train.py` (QLoRA, kayıp yalnızca cevapta, `--smoke-test`) ve `scripts/predict.py`; yerelde duman testi.

## 2026-11-24, görev 01 (GECTurk dönüştürme, alt küme, sızıntı) ve görev 02 (film yorumları)

- Yapılan:
  - `trgec.convert`: `parse_m2`, `apply_m2_edits`, `resolve_overlapping_edits`, `convert_gecturk`. 5 yeni test.
  - `scripts/prepare_data.py`: `convert` adımı GECTurk'ün dört dosyasını da çeviriyor; `sample` adımına GECTurk katmanlı alt kümesi (`sample_gecturk`) eklendi; `leakage` adımı yazıldı (üretilmiş dosyaları bağımsız tarar, çakışma varsa hata verir). Betik tek komutla baştan sona çalışıyor.
  - `scripts/show_examples.py`: elle kontrol için rastgele örnekleri, değişen kelimelerle birlikte yazdırır.
  - `scripts/evaluate.py`: film yorumları ERRANT-TR ile bağlandı. mT5 film yorumlarında çalıştırıldı.
  - `raporlar/01_veri_raporu.md` tamamlandı (GECTurk bölümü, kural tablosu, bütün çiftlerin sızıntı taraması, lisans özeti); `03_TEKNIK_NOTLAR.md` içine üç yeni tuzak eklendi.
- Ölçülen:
  - Dönüştürme doğrulaması: M2 düzenlemeleri uygulanınca hedef dosyasıyla birebir aynı cümle: train 96.919 / 96.919, dev 20.769 / 20.769, test 20.769 / 20.769.
  - GECTurk train havuzu: 96.919 satırdan 2 (değişmeyen düzenleme), 293 (yineleme), 100 (dev ile çakışan), 84 (test ile çakışan) atıldı; film yorumları, OSCAR testi ve tweetlerle çakışma 0. Havuz 96.440.
  - `gecturk_train_12k`: 6.000 hatalı + 6.000 hatasız, 9.513 düzenleme. Her kuraldan en az 50 cümle; istisnalar `rule_19` (3, hepsi) ve `rule_21` (38, hepsi).
  - Sızıntı taraması: 21 çiftin hepsinde 0. GECTurk dev'den tweetlerle aynı olan 1 cümle çıkarıldı (dev 20.768).
  - Film yorumlarında 3 cümlede örtüşen düzenleme var (6, 75, 125).
  - **mT5, film yorumları (ERRANT-TR):** P 30,6, R 23,3, F0.5 28,8 (güven aralığı 22,6–35,3); TP 53, FP 120, FN 174. Yayımlanan 30,0 / 22,5 / 28,1. Fark 0,7 puan. Üç yayımlanmış mT5 skoru da 1 puanın içinde yeniden üretildi.
  - Testler: 79 geçti.
- Karar ya da değişiklik:
  - Film yorumlarında örtüşen düzenlemeler için kural: en geniş olan, eşitse sonra yazılan uygulanır (yalnızca ERRANT-TR için üretilen hedef cümleyi etkiler; kullanıcıya bildirildi).
  - GECTurk dev'den test setiyle çakışan 1 cümle çıkarıldı.
- Sorun ya da açık soru:
  - **OSCAR hedefleri kısmen düzeltilmiş.** Bakılan üç örneğin üçünde hedefte hata kalmış (`ayni`, `yapiyordum`). OSCAR'dan üretilen hatasız örnekleri (K27) etkiler; kullanıcının elle kontrolünde sayılmalı.
  - GECTurk dev 20.768 cümle; pilotta tamamı mı, bir alt kümesi mi kullanılacak, görev 03'te kararlaştırılacak.
  - M2 scorer henüz bağlanmadı (etiketleyicinin 84,2'si ve GECTurk dev için gerekli).
  - Dünkü ve bugünkü değişiklikler commit edilmedi.
- Sıradaki adım: Kullanıcının elle kontrolü (`show_examples.py`). Görev 03'ün yerel kısmı: eğitim örneklerinin biçimi (`build_messages`, GECTurk için `extra.edits`), gerçek hedeflerle token uzunluğu ölçümü, `train.py` ve `predict.py`, yerel duman testi. Görev 02'den kalan: M2 scorer ve etiketleyici denemesi.

## 2026-11-24, görev 01 (GECTurk M2 dosyaları indirildi)

- Yapılan: Drive klasörü tarayıcıda açılıp içeriği doğrulandı (herkese açık). Kullanıcı onayıyla yedi dosya `data/raw/gecturk/` altına indirildi ve biçimleri incelendi; bulgular `raporlar/01_veri_kaynaklari.md` içinde ("İndirilen M2 dosyaları").
- Ölçülen:
  - Toplam 31,6 MB. Cümle sayıları HF kopyasıyla aynı: train 96.919, val 20.769, test 20.769, film yorumları 300.
  - Düzenleme sayısı: train 72.997, val 15.520, test 15.585, film yorumları 227. Hatasız cümle oranı dört dosyada da yaklaşık %50.
  - En seyrek kurallar (train): `rule_19` 3, `rule_21` 38, `rule_3` 66, `rule_2` 72.
- Karar ya da değişiklik: Yok. Düzeltme: "Drive komut satırından indirilemiyor" notu denenmeden yazılmıştı; raporda düzeltildi.
- Sorun ya da açık soru:
  - Dosya uzantısı `.txt`; `prepare_data.py` şu an `*.m2` arıyor, dönüştürücü yazılırken düzeltilecek.
  - Film yorumlarının hedef dosyası yok; hedefler düzenlemelerden üretilecek.
  - `min_per_rule: 50` iki kuralda (`rule_19`, `rule_21`) karşılanamıyor.
- Sıradaki adım: GECTurk dönüştürücüsü (M2 → ortak JSONL; düzenlemeleri uygulayıp hedef dosyasıyla karşılaştırarak doğrulama), 12 binlik katmanlı alt küme, GECTurk tarafının sızıntı taraması, film yorumlarının `evaluate.py`'ye bağlanması.

## 2026-11-24, görev 02 (karşılaştırma modu)

- Yapılan: Kullanıcı karşılaştırma modunun seçimini bana bıraktı. Ana skor `cs` olarak sabitlendi; `evaluate.py` sonuç dosyasına `primary_mode` yazıyor ve çıktıda ana modu `*` ile işaretliyor. `cse` hesaplanmaya devam ediyor.
- Ölçülen: Yeni ölçüm yok. Gerekçedeki sayılar: iki mod arasında tweetlerde 1, OSCAR testinde 2 düzenleme fark var.
- Karar ya da değişiklik: K15'e 2026-11-24 notu; `base.yaml` içinde `eval.errant_tr_mode: cs`.
- Sorun ya da açık soru: GECTurk M2 dosyaları hâlâ yok; D1 ve D2'nin GECTurk yarısı, GECTurk doğrulama seti ve film yorumları bunu bekliyor. Dünkü ve bugünkü değişiklikler commit edilmedi.
- Sıradaki adım: Görev 03'ün M2 dosyası gerektirmeyen yerel kısmı: eğitim örneklerinin biçimi (OSCAR tarafı), `train.py` ve `predict.py`, yerel duman testi. M2 dosyaları gelince görev 01'in kalanı (GECTurk dönüştürücüsü, 12 binlik alt küme, sızıntı taraması, elle kontrol).

## 2026-11-23, görev 02 (adım 2: mT5 çıkarımı ve Kapı A)

- Yapılan:
  - Kullanıcı `torch`, `transformers` ve `sentencepiece` bağımlılıklarını onayladı. Kuruldu: `torch` 2.15.0 (yalnızca CPU), `transformers` 5.24.0, `sentencepiece` 0.2.3. `requirements.txt` ve `KURULUM.md` güncellendi.
  - `src/trgec/seq2seq.py` (modelden bağımsız yardımcılar: uzunluğa göre gruplama, çıktıyı tek satıra indirme, sınıra takılma denetimi, koşu özeti) ve `scripts/predict_seq2seq.py` yazıldı. 5 yeni test.
  - `scripts/evaluate.py` içine `--hypothesis-field` eklendi (tahmin dosyasındaki başka bir alanı skorlamak için).
  - mT5 (`asimokby/Turkish-OSCAR-GEC-v0` @ `3467f87`) yerelde CPU ile çalıştırıldı: önce `oscar_dev` diliminden 5 cümle, sonra `oscar_dev` (500), tweetler (2.000) ve OSCAR testi (2.408). Kaggle gerekmedi.
- Ölçülen:
  - **Kapı A, tweetler:** P 84,6, R 61,1, F0.5 78,5 (güven aralığı 77,4–79,6); TP 3.836, FP 701, FN 2.441. Yayımlanan: 85,1 / 61,3 / 79,0. Fark 0,5 puan, ±2 ölçütü sağlandı. Ayar: açgözlü kod çözme, 256 token sınırı, `capitalize: turkish`, `keep_punct: "'’#"`.
  - `cs` ve `cse` modları tweetlerde 1 düzenleme farkla aynı (78,53 ve 78,51).
  - **İkincil kontrol, OSCAR testi:** P 68,5, R 30,8, F0.5 55,1 (güven aralığı 53,7–56,5); TP 6.890, FP 3.164, FN 15.476. Yayımlanan: 68,7 / 31,2 / 55,4. Fark 0,3 puan.
  - OSCAR testi, kaynak uzunluğuna göre: 48 token ve altı (1.441 cümle) F0.5 71,0; üstü (967 cümle) F0.5 43,6. Uzun cümlelerin 698'inde çıktı tam 48 tokenda bitti.
  - `oscar_dev` (hat denemesi; mT5 bu veriyi eğitimde gördü, skor şişkin): P 87,2, R 80,5, F0.5 85,8.
  - mT5, girdisi 48 tokenı aşan 186 `oscar_dev` cümlesinin 137'sinde çıktıyı tam 48 tokenda bitirdi. 48 tokenı aşan kaynak cümle: tweetlerde 3 (%0,1), OSCAR testinde 967 (%40,2).
  - Tokenizer'ın temizleme adımı tweet çıktılarının hiçbirini, `oscar_dev` çıktılarının 23'ünü değiştiriyor.
  - Hız (CPU): tweetlerde 1,25 cümle/sn, `oscar_dev` diliminde 0,45 cümle/sn.
  - Testler: 74 geçti.
- Karar ya da değişiklik: `base.yaml` içine `baselines.oscar_mt5` eklendi (`num_beams: 1`, `max_new_tokens: 256`). **Kapı A geçildi.** `eval.errant_tr_mode` hâlâ `null`; seçim kullanıcıya soruldu.
- Sorun ya da açık soru:
  - Karşılaştırma modu (`cs` ya da `cse`) kullanıcı kararını bekliyor.
  - Uzun satırlar: mT5 çıktıyı kesiyor ve yayımlanmış 55,4 bunu içeriyor. Bizim modellerimizin uzun satırları çıkarımda nasıl işleyeceği (cümlelere bölme) ve skorun uzunluk dilimlerine göre ayrı raporlanması görev 03'te kararlaştırılacak.
  - GECTurk M2 dosyaları hâlâ yok (film yorumları, M2 scorer, GECTurk alt kümesi bunu bekliyor).
  - Bugünkü değişiklikler commit edilmedi.
- Sıradaki adım: Karşılaştırma modunu sabitle; GECTurk M2 dosyaları gelince görev 01'in kalanı ve film yorumları; ardından görev 03 (eğitim örneklerinin biçimi, `train.py` ve `predict.py`, yerel duman testi).

## 2026-11-23, görev 02 (adım 7: `evaluate.py` ve Kapı A hazırlığı)

- Yapılan:
  - `src/trgec/evaluation.py` yazıldı: tweet dönüşümleri (`capitalize_first`, `strip_punctuation`, `tweet_protocol`), `align_predictions`, `summarize` (bootstrap ile), `clean_sentence_stats`. 6 test.
  - `scripts/evaluate.py` yazıldı: tahmin dosyası + set adı → `cs` ve `cse` modlarında P, R, F0.5, %95 güven aralığı, aşırı düzeltme oranı, cümle bazlı sayılar. `--baseline source|gold` ile akıl sağlığı koşusu yapıyor.
  - mT5 temel modelinin yapılandırması ve yazarların README'si incelendi; bulgular `raporlar/02_skorlayici.md` içinde.
- Ölçülen:
  - Tweetler: altın cümle hipotez olunca TP 6.277, FP 0, FN 0 (F0.5 = 1,0); kaynak cümle hipotez olunca TP 0, FN 6.277 (recall 0).
  - OSCAR test: kaynak cümle hipotez olunca TP 0, FN 22.366 (recall 0). Altın koşusu `check_errant_tr.py` ile daha önce yapıldı (F0.5 = 1,0).
  - `oscar_dev_clean`: kaynak cümle hipotez olunca hiçbir düzenleme yok; aşırı düzeltme oranı 0.
  - Tweet referansında 1.717 cümlede ilk kelimede düzenleme var; 55 altın cümle küçük `i` ya da `ı` ile başlıyor.
  - Yayımlanmış P ve R (OSCAR GEC mT5): tweetler 85,1 / 61,3; OSCAR test 68,7 / 31,2; film yorumları 30,0 / 22,5.
  - Testler: 69 geçti.
- Karar ya da değişiklik: `base.yaml` içine `eval.tweets` (`capitalize: turkish`, `keep_punct: "'’#"`) eklendi; bunlar varsayılan, Kapı A'da sabitlenecek.
- Sorun ya da açık soru:
  - mT5'in `generation_config.json` dosyasında üretim ayarı yok; ışın sayısı ve çıktı uzunluğu bilinmiyor. Kapı A'nın en büyük belirsizliği bu.
  - mT5'i çalıştırmak için `transformers`, `torch` ve `sentencepiece` gerekiyor; yerel deneme için kullanıcı onayı bekleniyor.
  - GECTurk M2 dosyaları hâlâ yok.
- Sıradaki adım: Bağımlılık onayı gelince mT5 çıkarım betiği (`scripts/predict_seq2seq.py`), yerelde 5 cümleyle deneme, ardından Kaggle'da üç test setinde çıkarım ve tweetlerde 79,0'ın yeniden üretilmesi.

## 2026-11-23, görev 02 (adım 6: ERRANT-TR kurulumu)

- Yapılan:
  - Kullanıcı `rapidfuzz` ve `zemberek-python` bağımlılıklarını onayladı. İkisi kuruldu; `zemberek-python`'ın çalışması için `setuptools<81` de gerekti ve eklendi.
  - ERRANT-TR `external/erranttr` altına `56cf0a3` sürümüyle klonlandı; iki kaynak dosya sürüm sayfasından indirildi. `external/` `.gitignore`'a eklendi; kurulum adımları `KURULUM.md` içinde.
  - `src/trgec/errant_tr.py` yazıldı: `load_annotator`, `parallel_to_m2`, `map_labels`, `sentence_counts`, `total_counts`. 9 test.
  - `scripts/check_errant_tr.py` yazıldı: resmi OSCAR M2 dosyasını yeniden üretip karşılaştırır.
  - Skorlayıcı raporu yazıldı (`raporlar/02_skorlayici.md`).
- Ölçülen:
  - Resmi OSCAR M2 dosyası altın kaynak ve altın düzeltmeden yeniden üretildi: 2.408 bloğun 2.408'i birebir aynı; `cs` ve `cse` modlarında TP 22.366, FP 0, FN 0.
  - Süre: 2.408 satır için 817 saniye. Zemberek'in yüklenmesi yaklaşık 7 saniye.
  - Testler: 63 geçti (yaklaşık 8 saniye; ERRANT-TR testleri Zemberek'i yüklüyor).
- Karar ya da değişiklik: Yeni bağımlılıklar: `rapidfuzz`, `zemberek-python`, `setuptools<81`. `base.yaml` içine `eval.errant_tr_dir` ve `eval.errant_tr_mode` (şimdilik `null`) eklendi.
- Sorun ya da açık soru:
  - Python 3.13'te `typing.io` kaldırıldığı için `antlr4` 4.8 doğrudan çalışmıyor; `trgec.errant_tr` içinde küçük bir yamayla çözüldü.
  - `map_labels.py` cümle metnini de değiştiriyor (`DAMACANA` → `SPELLMACANA`); resmi referans böyle üretildiği için davranış aynen korundu.
  - Karşılaştırma modu (`cs` ya da `cse`) bilinmiyor; Kapı A'da 79,0 ile sabitlenecek.
  - Tweetler için resmi M2 yok; referans bu araçla üretilecek.
  - GECTurk M2 dosyaları hâlâ yok (görev 01'in GECTurk kısmı ve film yorumları bunu bekliyor).
- Sıradaki adım: Kapı A hazırlığı: mT5 (`asimokby/Turkish-OSCAR-GEC-v0`) için çıkarım betiği, yerelde 5 cümleyle deneme, tweet dönüşümleri ve `scripts/evaluate.py`. Tam çıkarım Kaggle'da (eğitim değil, yalnızca temel modelin çalıştırılması).

## 2026-11-23, görev 01 (K27 oranı: %25)

- Yapılan: Kullanıcı hatasız örnek oranını %25 olarak seçti (önerilen %50 idi). `sampling.oscar_clean_ratio: 0.25` yapıldı; alt küme havuzlardan yeniden üretildi (veri yeniden okunmadı). K27, veri raporu, proje özeti ve iş planı güncellendi.
- Ölçülen:
  - `oscar_train_12k`: 9.000 hatalı + 3.000 hatasız; kaynak ve hedefi aynı olan 3.007. 9.000 hatalı satır, ilk seçilen 12.000'in ilk 9.000'i. `oscar_dev` (500) ve `oscar_dev_clean` (250) değişmedi.
  - 9.000 hatalı satırda 27.835 düzeltme. D2'nin hatasız oranı %37,5 (D1 %50).
  - 512 tokenı aşan örnek (12.750 örnekte): Qwen3 3, EuroLLM 2, Kanarya 1, diğerleri 0.
  - Dosyalar arasında ortak satır ve ortak cümle yok. Testler: 54 geçti.
- Karar ya da değişiklik: K27'de oran %25 olarak kesinleşti.
- Sorun ya da açık soru: D1 ve D2'nin hatasız oranı farklı (%50 ve %37,5); raporda sınırlılık. Diğer açık noktalar bir alttaki kayıtta.
- Sıradaki adım: M2 dosyaları gelince GECTurk dönüştürücüsü, katmanlı alt küme ve GECTurk tarafının sızıntı taraması. M2 gelmeden: görev 02'nin ilk adımı (ERRANT-TR kurulumu; iki yeni bağımlılık için onay gerekir).

## 2026-11-23, görev 01 (adım 5: OSCAR alt kümesine hatasız örnek, K27)

- Yapılan:
  - Kullanıcı OSCAR alt kümesine hatasız örnek eklenmesine karar verdi; K27 yazıldı.
  - `trgec.sampling` içine `Reservoir` sınıfı ve `mix_with_clean`, `trgec.convert` içine `oscar_clean_record` eklendi; 5 yeni test.
  - `prepare_data.py --step sample` iki havuzla çalışıyor: hatalı havuz (tohum 42) ve aynı akıştan bağımsız hatasız havuz (tohum 43). Havuzlar `data/processed/pools/` altında saklanıyor; oran değişince veri yeniden okunmuyor (`--refresh-pools` yeniden okutur).
  - Veri raporu güncellendi (`raporlar/01_veri_raporu.md`).
- Ölçülen:
  - Hatalı havuz 13.999, hatasız havuz 7.438 satır (60 satır iki havuzda da çıktığı için, 1 yineleme ve 1 aynı metin atıldı). Test setleriyle çakışma 0.
  - `oscar_train_12k`: 6.000 hatalı + 6.000 hatasız (kaynak ve hedefi aynı olan 6.003). `oscar_dev` 500 (önceki seçimle birebir aynı), `oscar_dev_clean` 250.
  - Eğitimdeki 6.000 hatalı satır, önceki 12.000'in ilk 6.000'i. Dosyalar arasında ortak satır ve ortak cümle yok.
  - Hata yoğunluğu: satır başına ortalama 3,1 düzeltme; 6.000 hatalı satırda 18.618 düzeltme.
  - 512 tokenı aşan örnek (12.750 örnekte): Qwen3 4, EuroLLM 2, Kanarya 1, diğerleri 0.
  - Testler: 54 geçti.
- Karar ya da değişiklik:
  - **K27 eklendi:** OSCAR alt kümesine hedef -> hedef hatasız örnek eklenir; toplam 12.000 kalır; doğrulama için `oscar_dev_clean` (250) ayrılır.
  - **Onay bekleyen:** oran. Uygulanan ve önerilen %50 (`sampling.oscar_clean_ratio: 0.5`).
- Sorun ya da açık soru:
  - OSCAR hedeflerinin nasıl üretildiği ve temizliği doğrulanmadı; hatasız örneklere elle kontrolde bakılmalı.
  - Organik hatalı satır 12.000'den 6.000'e indi; etkisi pilotta doğrulama setinde görülecek.
  - GECTurk M2 dosyaları hâlâ yok.
- Sıradaki adım: M2 dosyaları gelince GECTurk dönüştürücüsü, katmanlı alt küme ve GECTurk tarafının sızıntı taraması. M2 gelmeden: görev 02'nin ilk adımı (ERRANT-TR kurulumu; iki yeni bağımlılık için onay gerekir).

## 2026-11-23, görev 01 (adım 4–5, OSCAR kısmı)

- Yapılan:
  - Kullanıcı 70 kelime süzgecini ve 512 token tavanını onayladı; K11'e not düşüldü, `base.yaml` güncellendi (`oscar_max_words`, `oscar_reserve`, `max_len_ceiling`, `train_revision`).
  - `trgec.sampling.reservoir_sample`, `trgec.leakage` içinde `match_keys`, `split_by_overlap`, `dedupe_records`, `trgec.convert` içinde `oscar_row_problem` ve `oscar_train_record` yazıldı; 7 yeni test.
  - `scripts/prepare_data.py --step sample` yazıldı ve gerçek veride çalıştırıldı: `oscar_dev.jsonl` (500), `oscar_train_12k.jsonl` (12.000), kimlik listeleri ve istatistik dosyası üretildi.
  - Veri raporu taslağı yazıldı (`raporlar/01_veri_raporu.md`).
- Ölçülen:
  - `train.csv`: 1.737.696 satır; 79.929'u (%4,6) 70 kelimeden uzun; uygun 1.657.767.
  - Uygun satırlarda kaynak ve hedefi aynı olan 1.257 (%0,08). Alt kümede 9, `oscar_dev` içinde 0.
  - Seçilen 14.000 satırda 1 yineleme; OSCAR testi, tweetler ve film yorumlarıyla çakışma 0.
  - Alt kümede kaynak cümle ortalaması 21 kelime. 512 tokenı aşan örnek: Qwen3'te 4, EuroLLM'de 2, diğer tokenizer'larda 0 (12.500 örnek).
  - Testler: 49 geçti.
- Karar ya da değişiklik: K11'e 2026-11-23 notu (70 kelime, 512 tavan, kırpma yerine eleme).
- Sorun ya da açık soru:
  - **OSCAR'da hatasız cümle neredeyse yok (%0,08).** Olduğu gibi bırakmak mı, hatasız örnek eklemek mi; kullanıcıya soruldu.
  - Sızıntı taraması satırın tamamı üzerinden; bir test cümlesinin uzun bir eğitim satırının içinde geçmesi yakalanmaz.
  - "2,3 milyon çift" bütün veri setinin boyutu; `train.csv` 1,74 milyon satır.
  - GECTurk M2 dosyaları hâlâ yok.
- Sıradaki adım: M2 dosyaları gelince GECTurk dönüştürücüsü, katmanlı alt küme ve GECTurk tarafının sızıntı taraması. M2 gelmeden yapılabilecek iş: görev 02'nin ilk adımı (ERRANT-TR kurulumu; `rapidfuzz` ve `zemberek-python` bağımlılıkları için onay gerekir).

## 2026-11-23, görev 01 (adım 3: OSCAR testi ve tweet dönüştürücüleri) ve kapsam kararı

- Yapılan:
  - Kullanıcı yeni çerçeveyi ve sekiz adaylık listeyi onayladı. Kararlara işlendi: K1, K3, K5, K6, K7 ve K20'ye tarihli not; yeni K25 (sınırlı keşif fazı) ve K26 (verimlilik ve görülmemiş alan ölçümleri).
  - `00_PROJE_OZETI.md`, `02_IS_PLANI.md`, `CLAUDE.md` ve `configs/base.yaml` (`model.candidates`, sekiz kimlik) yeni çerçeveye uyduruldu.
  - "512 token yeter mi?" sorusu ölçüldü (`raporlar/03_model_adaylari.md`, bölüm 10).
  - `src/trgec/convert.py` yazıldı: `convert_oscar_test`, `convert_tweets`, `m2_edit_types`, `split_lines`, `write_jsonl`, `read_jsonl`. `scripts/prepare_data.py --step convert` bunları çağırıyor.
  - `tests/test_convert.py`: sahte veriyle 11 test.
- Ölçülen:
  - Dönüştürme: `oscar_test.jsonl` 2.408 örnek (26'sında kaynak ve hedef aynı), `tweets.jsonl` 2.000 örnek (89'unda aynı). İkisi de adım 2'deki sayılarla tutuyor.
  - OSCAR M2: `noop` dışında 22.366 düzenleme (22.392 − 26 `noop`). 4 satırda M2'de düzenleme yok ama `inp` ile `ref` metni farklı.
  - Tweetler: `|` işareti kaynak sütunda 85, hedef sütunda 425 satırda; ikisinden birinde 489 satırda. (13 Kasım kaydındaki 425 yalnızca hedef sütunuydu.) Dosyada 127 alanda baş ya da sonda boşluk, 11 alanda çift boşluk vardı; temizlendi. 47 birleşen nokta yerinde.
  - Token uzunluğu: 512 sınırını GECTurk örneklerinin %0,03–0,10'u, OSCAR örneklerinin %1,1–3,0'ı aşıyor. OSCAR'ın ilk 20.000 satırının en uzunu 15.475 kelime. Kaynak cümle 70 kelimeyle sınırlanınca satırların %95,5'i kalıyor ve 512'yi aşan en fazla %0,05.
  - Çıkarım: OSCAR test girdilerinde en uzun satır 453 kelime, 33 satır 100 kelimeden uzun.
  - Temel modeller: etiketleyici yaklaşık 110M (BERT-base), mT5-base 582M parametre.
  - Testler: 42 geçti.
- Karar ya da değişiklik:
  - K1, K3, K5, K6, K7, K20 notları; K25 ve K26 eklendi (kullanıcı onayıyla).
  - Tweetlerde boşluk temizliği (baş, son, çift boşluk) dönüştürücüde yapılıyor; harf ve işaretlere dokunulmuyor. OSCAR metni olduğu gibi bırakıldı.
  - **Onay bekleyen öneri:** OSCAR alt kümesinde kaynak cümle en fazla 70 kelime; pilotta bütün adaylar için 512 token tavanı; aşan örnek kırpılmaz, elenir. Nihai `max_len` Kapı B'de K11'e göre.
- Sorun ya da açık soru:
  - GECTurk M2 dosyaları hâlâ `data/raw/gecturk/` altında değil; GECTurk dönüştürücüsü ve adım 4–5'in GECTurk kısmı bunu bekliyor.
  - Uzun test satırlarının çıkarımda cümlelere bölünmesi tasarlanmadı (görev 02 ve 03).
  - 512 tokenda tepe GPU belleği ölçülmedi (pilot).
  - Hatasız metin için kaynak (K26) seçilmedi.
  - Proje haritası (kullanıcının `Downloads` klasöründeki HTML) 16 Kasım durumunu gösteriyor; yeni çerçeveyle güncellenmedi.
- Sıradaki adım: Görev 01, adım 4–5'in OSCAR kısmı: `oscar_dev` (500) ve OSCAR 12 bin alt kümesi (70 kelime süzgeci onaylanırsa onunla), ardından sızıntı taraması. M2 dosyaları gelince GECTurk dönüştürücüsü.

## 2026-11-21, görev 03 (adım 1, genişletildi: model adayları)

- Yapılan:
  - Danışman projeyi uygun gördü ve denenecek model sayısının artırılabileceğini söyledi. Bunun üzerine aday listesi genişletildi (`raporlar/03_model_adaylari.md`).
  - Türkçeye özel ve çok dilli 0,2B–9B açık modeller araştırıldı. Kimlik, parametre ve lisanslar HF API ile doğrulandı.
  - Yaklaşık 20 tokenizer, GECTurk ve OSCAR eğitim örneklerinde ölçüldü. Ölçüm, proje bağımlılıklarına dokunmadan geçici bir ortamda yapıldı.
  - T4 hız ölçümlerinden bir süre modeli kuruldu; her aday için mini, D1 ve D2 süreleri tahmin edildi.
- Ölçülen:
  - Kelime başına token: Kumru 1,81, Qwen3 2,74, EuroLLM 2,37, Türkçe GPT-2 1,68.
  - Eğitim örneği ortalaması (GECTurk / OSCAR): Kumru 136 / 140, Qwen3 186 / 212.
  - Dolgu: rastgele batch'te %95–243, uzunluğa göre gruplu batch'te %16–29.
  - Tahmini süre, tek T4: Kumru D1 1,8 sa, D2 3,7 sa; Qwen3-1.7B D1 2,2 sa, D2 4,7 sa. İki T4 DDP ile D2 2,0 ve 2,6 sa.
- Karar ya da değişiklik:
  - **K24 eklendi:** ortak biçimde doğal metin saklanır (kullanıcı onayladı).
  - Kullanıcı pilot aday listesinin genişletilmesini istedi. Sekiz adaylık son liste raporun 9. bölümünde; kullanıcının kesinleştirmesi bekleniyor. Kesinleşince K7 ve K6'ya not düşülecek, `base.yaml` içindeki `model.candidates` güncellenecek.
  - Ana deney (D1 ve D2) tek modelle kalıyor; ikinci modelle tam deney pilot sonucuna göre konuşulacak.
- Sorun ya da açık soru:
  - Süre tahmini 0,6–1,5 kat belirsiz; pilotta ölçülecek.
  - İlk taslakta hız çapası yanlış bağlanmıştı (süreler yaklaşık 1,5 kat kısa çıkıyordu); raporda düzeltildi ve not düşüldü.
  - Kaggle kotasının oturum başına mı GPU başına mı düştüğü doğrulanmadı.
  - GECTurk M2 dosyaları hâlâ `data/raw/gecturk/` altında değil.
  - **Çerçeve sorusu açık:** kullanıcı projeyi danışmana "daha hafif modeller ve nicemlemeyle eski sonuçlara yaklaşmak" olarak anlatmış. Ölçülen boyutlar: etiketleyici yaklaşık 110M (BERT-base), mT5-base 582M; ana adaylarımız 1,7–2,4B. Yani parametre sayısıyla temel modellerden hafif değiliz. Seçenekler kullanıcıya sunuldu (verimlilik tablosu, boyut merdiveni, nicemleme kıyası); karar bekleniyor.
  - **Kapsam sorusu açık:** kullanıcı hedefi "hafif modellerle, üç test setinin ötesinde genellenebilir bir düzeltici" olarak genişletmek ve tekniklerde serbest kalmak istiyor. Önerilen yapı: çekirdek (D1 ve D2, üç test seti) korunur; yalnızca doğrulama setlerinde çalışan, süresi ve teknik sayısı sınırlı bir keşif fazı ve görülmemiş alanlarda ek ölçümler eklenir. Bu K1, K3, K5 ve K20'yi (Kapı B) etkiler; kullanıcı kararı bekleniyor, hiçbir karar değiştirilmedi.
- Sıradaki adım: Liste kesinleşince görev 01, adım 3: OSCAR testi ve tweet dönüştürücüleri, sahte veriyle testleri.

## 2026-11-13, görev 01 (adım 2: tokenizasyon)

- Yapılan:
  - OSCAR testi (`04cd0c3`) ve tweet seti (`06e9476`) sabit sürümle `data/raw/` altına indirildi.
  - İki setin ve ERRANT-TR'nin tokenizasyonu incelendi (`raporlar/01_tokenizasyon.md`).
  - `normalize_for_match` içindeki birleşen nokta hatası düzeltildi ve testi eklendi.
- Ölçülen:
  - OSCAR testi: 2.408 satır; `inp` ile M2 `S` satırı yalnızca 291 satırda aynı; hatasız satır 26 (%1,1); ortalama 25 token.
  - Tweetler: 2.000 satır; hatasız 89 (%4,5); `|` işaretli 425 satır; 47 birleşen nokta.
  - Testler: 31 geçti.
- Karar ya da değişiklik: Yok. "Ortak biçimde doğal metin" önerisi kullanıcı onayı bekliyor.
- Sorun ya da açık soru:
  - GECTurk M2 dosyaları hâlâ yok; M2 `S` satırı ile parquet tokenlarının birebir aynı olduğu doğrulanamadı.
  - Ersoy ve Yıldız'ın tweet çıktılarından hangi noktalamayı sildiği belirsiz (Kapı A riski).
- Sıradaki adım: Onay gelince adım 3: OSCAR testi ve tweet dönüştürücüleri, sahte veriyle testleri.

## 2026-11-13, görev 01 (karar güncellemesi)

- Yapılan:
  - K15'e tarihli not eklendi.
  - `00_PROJE_OZETI.md` içindeki temel tablo ve başarı ölçütü düzeltildi.
  - Görev 02, iş planı ve `base.yaml` (`eval.scorers`) yeni karara uyduruldu.
  - Koddaki yanlış karar atıfları düzeltildi (K12 → K15, K13 → K14/K16).
- Ölçülen: Yok.
- Karar ya da değişiklik:
  - K15 güncellendi: OSCAR ve tweetlerde ERRANT-TR; film yorumlarında M2 scorer ve ERRANT-TR birlikte.
  - OSCAR testinde hedef 55,4 (OSCAR GEC mT5). 62,8 (GPT GEC mT5) ikinci referans.
- Sorun ya da açık soru: GECTurk M2 dosyaları henüz `data/raw/gecturk/` altında değil.
- Sıradaki adım: Görev 01, adım 2 (tokenizasyon) ve adım 3 (dönüştürücüler).

## 2026-11-12, görev 01 (adım 1)

- Yapılan:
  - Depo GitHub'a gönderildi (`aesivrioglu/Turkish_Correction_Model`, herkese açık) ve `.venv` kuruldu.
  - Üç belgedeki bozuk UTF-8 baytı düzeltildi.
  - Doğrulanmamış varsayımlar araştırıldı (`raporlar/00_varsayim_dogrulama.md`).
  - Veri kaynakları incelendi (`raporlar/01_veri_kaynaklari.md`).
  - `base.yaml` içindeki Kumru kimliği `vngrs/Kumru-2B` olarak düzeltildi ve OSCAR kimlikleri eklendi.
  - `05_KAYNAKLAR.md` içindeki yayımlanmış sayılar düzeltildi.
- Ölçülen:
  - GECTurk ayrımları: train 96.919, dev 20.769, test 20.769, movie_reviews 300. Hepsinde hatasız cümle oranı yaklaşık %50.
  - GECTurk'te tek başına noktalama tokenı oranı %0,16; noktalama kelimeye bitişik.
  - Train'de 19 numaralı kuraldan 3, 21 numaralı kuraldan 38 cümle var.
  - Tweet seti 2.000 satır.
- Karar ya da değişiklik: Yok. Önerilen değişiklikler kullanıcı onayı bekliyor (K15 skorlayıcı, OSCAR testinde 55,4 ile 62,8).
- Sorun ya da açık soru:
  - GECTurk'ün HF kopyasında hedef cümle yok; M2 dosyaları Google Drive'da ve kullanıcının indirmesi gerekiyor.
  - Tweet seti ve mT5 ağırlıklarının lisansı yok; danışmana sorulacak.
  - Kumru 2,38B parametreyle K6'daki aralığın biraz üstünde.
- Sıradaki adım: M2 dosyaları `data/raw/gecturk/` altına gelince görev 01, adım 2 (tokenizasyon) ve adım 3 (dönüştürücüler).

## 2026-11-12, başlangıç

- Yapılan: Depo iskeleti, yardımcı modüller (`src/trgec`) ve testler hazırlandı. Kararlar, plan ve teknik notlar `docs/` altına yazıldı.
- Ölçülen: 30 test geçiyor. GECTurk veri klasörü 12 MB.
- Karar ya da değişiklik: K1 ile K23 arası kararlar kaydedildi.
- Sorun ya da açık soru: Danışman onayı ve teslim tarihleri bekleniyor.
- Sıradaki adım: `gorevler/01_veri_hazirligi.md`
