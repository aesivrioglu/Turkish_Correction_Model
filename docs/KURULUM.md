# Kurulum

Bu adımlar bir kez yapılır. Gereken: Git, Python 3.10 ya da üstü, bir GitHub hesabı ve Claude Code için bir Claude aboneliği (Pro, Max, Team ya da Enterprise) veya Claude Console hesabı.

## 1. GitHub deposunu oluştur

GitHub'da yeni bir depo aç:

- Ad: `turkce-yazim-duzeltici`
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
git remote add origin https://github.com/KULLANICI_ADIN/turkce-yazim-duzeltici.git
git push -u origin main
```

## 3. Python ortamı

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest -q                        # 30 test geçmeli
```

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
