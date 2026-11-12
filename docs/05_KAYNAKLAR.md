# Kaynaklar

## Veri ve modeller

| Kaynak | Ne için | Bağlantı | Durum |
| --- | --- | --- | --- |
| GECTurk veri seti | Eğitim, film yorumları testi | https://huggingface.co/datasets/GGLab/GECTurk | Doğrulandı: train, dev, test, movie_reviews; parquet; Apache-2.0; toplam 12 MB |
| GECTurk deposu | M2 dosyaları, M2 scorer, etiketleyici kodu, BOUN de/da verisi | https://github.com/GGLAB-KU/gecturk | İncelendi |
| GECTurk etiketleyicisi | 2023 temel modeli | Hugging Face: GGLab/gec-tr-seq-tagger | Görev 02'de çalıştırılacak |
| Turkish-GEC deposu | OSCAR GEC eğitim ve test verisi, mT5 modelleri, yazım sözlüğü | https://github.com/asimokby/Turkish-GEC | Hugging Face kimlikleri görev 01'de README'den alınacak |
| #Turki$hTweets | Tweet test seti | https://github.com/atubakoksal/annotated_tweets | Görev 01'de doğrulanacak |
| TDK yazım kuralları | Kural kartları | https://www.tdk.gov.tr/kategori/icerik/yazim-kurallari/ | Görev 05'te kullanılacak |

## Makaleler

- Kara, A., Marouf Sofian, F., Bond, A., Şahin, G. G. (2023). GECTurk: Grammatical Error Correction and Detection Dataset for Turkish. Findings of IJCNLP-AACL 2023. https://arxiv.org/pdf/2309.11346
- Ersoy, A., Yıldız, O. T. (2024). Organic Data-Driven Approach for Turkish Grammatical Error Correction and LLMs. https://arxiv.org/abs/2405.15320v1
- GECTurk WEB: An Explainable Online Platform for Turkish Grammatical Error Detection and Correction (2024). https://arxiv.org/html/2410.12350v1
- Uz, H., Eryiğit, G. (2023). Towards Automatic Grammatical Error Type Classification for Turkish (ERRANT-TR). EACL SRW. https://aclanthology.org/2023.eacl-srw.14
- Koksal, A. T. vd. (2020). #Turki$hTweets: A Benchmark Dataset for Turkish Text Correction. Findings of EMNLP. https://preview.aclanthology.org/new-sigs/2020.findings-emnlp.374/
- Güzel Turhan, C. (2025). Leveraging large language models for spelling correction in Turkish. PeerJ Computer Science. https://peerj.com/articles/cs-2889/
- Türker, T., Eryiğit, G. (2026). Instruction-Following LLMs for Grammatical Error Correction. BEA 2026. https://aclanthology.org/2026.bea-1.17/
- Dettmers, T. vd. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314

## Yayımlanmış temel sayılar

Bu sayılar yukarıdaki makalelerden ve depolardan alınmıştır. Görev 02'de yeniden üretilir.

| Model | Test seti | F0.5 | Kaynak |
| --- | --- | --- | --- |
| GECTurk etiketleyicisi | GECTurk sentetik test | 0,98 | Kara vd. |
| GECTurk etiketleyicisi | Film yorumları | 0,84 (84,2) | Kara vd.; Turkish-GEC deposu |
| GECTurk etiketleyicisi | OSCAR GEC test | 14,7 | Turkish-GEC deposu |
| mT5, OSCAR GEC | Film yorumları | 28,1 | Ersoy ve Yıldız, Tablo 4 |
| mT5, OSCAR GEC | OSCAR GEC test | 55,4 | Ersoy ve Yıldız, Tablo 4 |
| mT5, OSCAR GEC | Tweetler | 79,0 | Ersoy ve Yıldız, Tablo 4 |
| mT5, GPT GEC | OSCAR GEC test | 62,8 | Ersoy ve Yıldız, Tablo 4 |

2026-11-12 notu: Bu tabloda daha önce OSCAR GEC (mT5) için OSCAR testinde 62,8 yazıyordu; o sayı GPT GEC (mT5) modeline ait. Ersoy ve Yıldız'ın bütün sayıları ERRANT-TR ile hesaplanmıştır; etiketleyicinin film yorumlarındaki 84,2'si GECTurk makalesinden (M2 scorer). Ayrıntı: `raporlar/00_varsayim_dogrulama.md`.

## Araçlar

- Claude Code belgeleri: https://code.claude.com/docs/en/quickstart
- Kaggle kotası hakkında: https://aimultiple.com/free-cloud-gpu (haftada yaklaşık 30 GPU saati, oturum başına 12 saat; resmi olarak sabit değil)
