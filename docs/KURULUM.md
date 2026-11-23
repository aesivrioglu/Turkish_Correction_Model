# Kurulum

Bu adımlar bir kez yapılır. Gereken: Git, Python 3.10 ya da üstü, bir GitHub hesabı ve Claude Code için bir Claude aboneliği (Pro, Max, Team ya da Enterprise) veya Claude Console hesabı.

## 1. GitHub deposunu oluştur

GitHub'da yeni bir depo aç:

- Ad: `Turkish_Correction_Model` (yerel klasörün adı `turkce-yazim-duzeltici` olarak kalabilir)
- Görünürlük: Private (lisans kontrolleri bitene kadar)
- README, .gitignore ve lisans **ekleme**; bunlar bu klasörde zaten var.

## 2. Kodu gönder

Zip'i açtığın klasörde:

```bash
cd turkce-yazim-duzeltici
git init
git add .
git commit -m "Proje iskeleti, kararlar ve yardımcı modüller"
git branch -M main
git remote add origin https://github.com/aesivrioglu/Turkish_Correction_Model.git
git push -u origin main
```

## 3. Python ortamı

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # yalnızca CPU sürümü (yaklaşık 250 MB)
pip install -r requirements.txt
pip install -e .
pytest -q                        # bütün testler geçmeli
```

### ERRANT-TR (değerlendirme için)

OSCAR testi, tweetler ve film yorumları ERRANT-TR ile skorlanır (K15). Araç paketlenmemiş olduğu için depoya girmez; `external/` altına elle kurulur:

```bash
mkdir -p external && cd external
git clone https://github.com/harunuz/erranttr.git
cd erranttr && git checkout 56cf0a3
cd erranttr/tr/resources
curl -L -O https://github.com/harunuz/erranttr/releases/download/v0.1.0/tr_TR.aff
curl -L -O https://github.com/harunuz/erranttr/releases/download/v0.1.0/TS_Corpus_Turkish_Word_List.txt.zip
unzip TS_Corpus_Turkish_Word_List.txt.zip
```

Notlar:

- Bağımlılıklar `requirements.txt` içinde: `rapidfuzz`, `zemberek-python` ve `setuptools<81`.
- `zemberek-python`, `antlr4-python3-runtime` 4.8 sürümüne bağlıdır ve bu sürüm Python 3.13'te kaldırılan `typing.io` modülünü ister. `trgec.errant_tr` bunu küçük bir yamayla çözer; ayrı bir Python sürümü gerekmez.
- Kurulum yoksa ERRANT-TR testleri atlanır, diğer testler çalışır.
- Kurulumun doğruluğu: `tests/test_errant_tr.py` geçmeli. Ayrıntı `docs/raporlar/02_skorlayici.md` içinde.

## 4. Claude Code

macOS, Linux ya da WSL:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Windows PowerShell:

```powershell
irm https://claude.ai/install.ps1 | iex
```

Sonra proje klasöründe:

```bash
claude
```

İlk çalıştırmada giriş yapman istenir. Güncel kurulum seçenekleri: https://code.claude.com/docs/en/quickstart

## 5. İlk görev

Claude Code'a şunu yaz:

```
CLAUDE.md dosyasını ve docs/ altındaki 00, 01 ve 03 numaralı belgeleri oku.
Sonra docs/gorevler/01_veri_hazirligi.md görevini bana kısaca özetle ve
1. adımdan (kaynakları doğrula) başla. Her adımın sonunda dur, ne bulduğunu
anlat ve devam etmeden önce onayımı bekle.
```

## 6. Kaggle (2. haftaya kadar)

`docs/04_ORTAM_VE_IS_AKISI.md` içindeki "Kaggle kurulumu" bölümünü izle.
