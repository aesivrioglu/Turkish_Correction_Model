# turkce-yazim-duzeltici

Türkçe metindeki yazım ve dilbilgisi hatalarını bulan, düzelten ve ihlal edilen TDK kuralını gösteren küçük bir dil modeli ve web demosu. Bitirme (capstone) projesi.

## Ne yapıyor?

Küçük bir açık LLM, QLoRA ile iki farklı veriyle eğitilir ve yayımlanmış iki modelle üç gerçek test setinde karşılaştırılır.

| Sistem | Eğitim | Kaynak |
| --- | --- | --- |
| BERTurk dizi etiketleyici (2023) | Hazır ağırlıklar | GECTurk |
| mT5 (2024) | Hazır ağırlıklar | Ersoy ve Yıldız |
| Bizim model, D1 | Yalnızca GECTurk | Bu proje |
| Bizim model, D2 | GECTurk + OSCAR GEC | Bu proje |

Test setleri: GECTurk film yorumları (300 cümle), OSCAR GEC test (2.400 cümle), #Turki$hTweets (2.000 tweet).

## Kurulum

```bash
git clone https://github.com/KULLANICI_ADIN/turkce-yazim-duzeltici.git
cd turkce-yazim-duzeltici
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

## Nerede ne çalışır?

Kod yerelde yazılır ve test edilir. Kaggle yalnızca GPU gereken işleri (eğitim ve model çıkarımı) çalıştırır. Ayrıntı: [docs/04_ORTAM_VE_IS_AKISI.md](docs/04_ORTAM_VE_IS_AKISI.md)

## Belgeler

- [Proje özeti](docs/00_PROJE_OZETI.md)
- [Kararlar ve nedenleri](docs/01_KARARLAR.md)
- [İş planı ve iş yükü](docs/02_IS_PLANI.md)
- [Teknik notlar ve tuzaklar](docs/03_TEKNIK_NOTLAR.md)
- [Ortam ve iş akışı](docs/04_ORTAM_VE_IS_AKISI.md)
- [Kaynaklar](docs/05_KAYNAKLAR.md)
- [Kurulum](docs/KURULUM.md)
- [Görevler](docs/gorevler/) ve [günlük](docs/GUNLUK.md)

## Durum

İskelet ve yardımcı modüller hazır. Sıradaki iş: veri hazırlığı (görev 01).
