# Ortam ve iş akışı

Kod yerelde Claude Code ile yazılır ve test edilir. Kaggle yalnızca GPU gereken işleri çalıştırır. Depo tek doğruluk kaynağıdır.

## Nerede ne yapılır?

| İş | Nerede | Neden |
| --- | --- | --- |
| Veri dönüştürme, sızıntı taraması, örnekleme | Yerel | CPU yeter; veri küçük |
| Değerlendirme, bootstrap, analiz | Yerel | CPU yeter |
| Eğitim ve çıkarım kodunun yazımı, duman testi | Yerel | Hatalar GPU kotası harcanmadan yakalanır |
| mT5 ve etiketleyici çıkarımı | Kaggle | GPU ile dakikalar sürer |
| Mini eğitimler, D1 ve D2, model çıkarımı | Kaggle | GPU gerekir |
| Kural kartları, demo, rapor | Yerel | GPU gerekmez |
| Plan, kararlar, sonuçların yorumu | Claude sohbet projesi | Plan belgesi orada |

## Neden her şey Kaggle'da değil?

- Claude Code Kaggle'ın tarayıcı editöründe çalışmaz.
- Not defterlerinde gizli durum vardır: hücreler sırasız çalışır, eski değişkenler kalır. Sonuçlar yeniden üretilemez hale gelir.
- İşin büyük kısmı GPU istemez; kota boşa harcanır.

## Bir deneyin yolculuğu

1. Yerelde kodu yaz ya da değiştir. `pytest -q` geçsin.
2. Eğitim kodu değiştiyse `python scripts/train.py --config ... --smoke-test` yerelde geçsin.
3. Commit at, GitHub'a gönder.
4. Kaggle'da `kaggle/calistirici.ipynb` not defterinde `RUNS` listesini ayarla, "Save Version" ile toplu çalıştır.
5. Betik bitince tahminleri, özetleri ve günlükleri tek bir zip'e koyar (pilotta `pilot_outputs.zip`; not defterinin Output bölümünden indirilir). Ana eğitimlerde adaptör ayrıca Hugging Face Hub'daki özel depoya yüklenir (`train.py --push-to-hub`).
6. Zip'i yerelde `outputs/kaggle/` altına aç, `python scripts/evaluate.py ...` ile değerlendir.
7. Sonucu ve gözlemleri `docs/GUNLUK.md` dosyasına yaz.

## Kaggle kurulumu (bir kez)

1. Kaggle hesabını telefonla doğrula. İnternet erişimi ve GPU için gereklidir.
2. Hugging Face'te yazma yetkili bir erişim anahtarı oluştur.
3. Kaggle not defterinde Add-ons > Secrets altına `HF_TOKEN` ekle (pilotta şart değil; adaptör Hub'a yüklenecekse gerekir). Depo herkese açık olduğu için `GITHUB_TOKEN` gerekmez.
4. `kaggle/calistirici.ipynb` dosyasını Kaggle'a yükle (File > Import Notebook).
5. Sağ panelde Accelerator = GPU T4 x2, Internet = On yap.
6. Not defterindeki `GITHUB_USER` (`aesivrioglu`) ve `REPO` (`Turkish_Correction_Model`) değerlerinin GitHub'daki depoyla aynı olduğunu kontrol et.
7. **Veri seti (2026-11-24):** `data/processed` depoya girmediği için eğitim ve doğrulama dosyaları özel bir Kaggle veri seti olarak yüklenir. `outputs/kaggle_dataset/trgec-pilot-veri.zip` (2,4 MB) dosyasını kaggle.com > Datasets > New Dataset ile yükle, görünürlüğü **Private** bırak, adını `trgec-pilot-veri` koy. Sonra not defterinde Input > Add Input ile ekle. Pakette yalnızca eğitim ve doğrulama dosyaları var (`gecturk_train_12k`, `oscar_train_12k`, `gecturk_dev_2k`, `oscar_dev`, `oscar_dev_clean`); test setleri bilerek konmadı.

## Duman testi nedir?

Eğitim betiğinin 8 örnek ve 2 adımla, 4-bit nicemleme kapalı olarak CPU'da çalıştırılmasıdır. Amaç modeli eğitmek değil; veri yükleme, biçimlendirme, kayıp hesabı ve kaydetme adımlarının hatasız çalıştığını görmektir. 4-bit nicemleme CUDA ister, bu yüzden yerelde kapatılır.

## Gizli bilgiler

- `HF_TOKEN` yerelde `.env` dosyasında, Kaggle'da Secrets altında durur.
- `.env` ve `kaggle.json` depoya girmez (`.gitignore` içinde).
- Anahtar yanlışlıkla commit edilirse hemen iptal edilip yenisi oluşturulur.

## İsteğe bağlı: Kaggle komut satırı aracı

`kaggle kernels push` ile not defteri yerelden başlatılabilir, `kaggle kernels output` ile çıktılar geri çekilebilir. Böylece Claude Code GPU'daki hatayı da görebilir. İlk haftalarda şart değildir; eğitim aşamasında zaman kazandırır.

**2026-11-24:** Araç kullanıcının bilgisayarında kurulu ve hesabına bağlı; pilot bu yolla yürütülüyor. Adımlar:

1. Veri seti: `outputs/kaggle_dataset/trgec-pilot-veri/` klasöründen `kaggle datasets create -p . -t` (özel; `-t` JSONL dosyalarının CSV'ye çevrilmesini önler). Klasörün içinden çalıştırılır; Windows'ta göreli yol hata veriyor.
2. Not defteri: `outputs/kaggle_kernel/` klasöründe `calistirici.ipynb` kopyası ve `kernel-metadata.json` (özel, GPU ve internet açık, veri seti bağlı). `kaggle kernels push -p . --accelerator NvidiaTeslaT4 -t <saniye>` koşuyu başlatır; `-t` süre sınırıdır.
3. `kaggle kernels status <kullanıcı>/trgec-pilot` ile durum, `kaggle kernels output <kullanıcı>/trgec-pilot -p <klasör>` ile çıktı ve günlük.

`kernel-metadata.json` ve `dataset-metadata.json` Kaggle kullanıcı adını içerdiği için depoya konmadı (`outputs/` altında).

## İki Claude, iki rol

- **Claude sohbet projesi:** Plan, kararlar, sonuçların yorumu, rapor taslağı. Kodu görmez.
- **Claude Code:** Depodaki kodu yazar, çalıştırır, test eder. `CLAUDE.md` ve `docs/` üzerinden projeyi bilir.

Bir karar sohbet projesinde değişirse `docs/01_KARARLAR.md` dosyasına işlenir; Claude Code oradan öğrenir.
