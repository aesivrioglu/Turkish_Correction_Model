# Proje özeti

## Amaç

Türkçe metindeki yazım ve dilbilgisi hatalarını bulan, düzelten ve hangi TDK kuralına aykırı olduğunu gösteren bir model ile web demosu geliştirmek.

## Araştırma soruları

1. **AS1, performans:** QLoRA ile ince ayarlanmış tek bir küçük LLM, üç gerçek test setinde 2023 etiketleyicisini ve 2024 mT5 modelini F0.5 cinsinden geçiyor mu?
2. **AS2, veri:** Eğitime organik yazım hataları eklemek sentetikten gerçeğe geçişi nasıl etkiliyor, hangi kurallarda yine de yetmiyor?

Aşırı düzeltme oranı ayrı bir soru değildir; her sonuçla birlikte raporlanır.

## Neden bu soru?

Bugün yayımlanmış hiçbir model üç gerçek test setinde birden iyi değil.

| Model | Film yorumları | OSCAR GEC test | Tweetler |
| --- | --- | --- | --- |
| GECTurk etiketleyicisi (2023) | 84,2 | 14,7 | GECTurk modelleri 42,3 ile 44,5 arasında |
| mT5, OSCAR GEC ile eğitilmiş (2024) | 28,1 | 55,4 | 79,0 |
| mT5, GPT GEC ile eğitilmiş (2024) | 37,6 | 62,8 | 75,8 |

Değerler F0.5'tir ve yayımlanmış çalışmalardan alınmıştır (bkz. `05_KAYNAKLAR.md`). Film yorumlarında etiketleyicinin skoru M2 scorer ile, diğer bütün sayılar ERRANT-TR ile hesaplanmıştır (K15). Etiketleyici 25 kurala sıkışıp sıradan yazım hatalarını göremiyor; mT5 ise kural tabanlı hataları kaçırıyor. Üç sette de dengeli tek bir model projenin ana iddiasıdır.

## Kapsam

- **İçinde:** GECTurk'ün 25 TDK kuralı (de/da, ki, mi, birleşik kelimeler, büyük harf vb.) ve OSCAR GEC'teki sıradan yazım hataları.
- **Dışında:** Anlam, üslup ve akıcılık düzeltmesi; cümleyi yeniden yazma.

## Yöntem, tek paragrafta

1 ile 2 milyar parametre arasında tek bir decoder LLM seçilir ve QLoRA ile iki kez eğitilir: D1 yalnızca GECTurk alt kümesiyle, D2 aynı alt küme artı OSCAR GEC alt kümesiyle. Model düzeltilmiş cümleyi ve her düzeltmenin kural etiketini üretir. Açıklama, kural etiketine bağlı sabit bir kural kartından gelir; model açıklama metni üretmez. Çıktı cümleyi fazla değiştirdiyse bir emniyet kemeri orijinal cümleyi korur.

## Teslim edilecekler

- Eğitilmiş LoRA adaptörleri (Hugging Face Hub)
- Eğitim ve değerlendirme kodu (bu depo)
- Üç sette karşılaştırma ve kural bazında analiz içeren rapor
- Hataların altını çizip kural kartını gösteren Gradio demosu

## Başarı ölçütleri (danışmanla netleştirilecek)

- Film yorumlarında etiketleyicinin 84,2'lik F0.5 skoruna ulaşmak ya da geçmek
- OSCAR ve tweet setlerinde OSCAR GEC ile eğitilmiş mT5'in 55,4 ve 79,0'lık skorlarına yaklaşmak. OSCAR testinde en yüksek yayımlanmış sonuç olan GPT GEC (mT5) 62,8 ikinci referans olarak raporlanır.
- Hatasız cümlelerde gereksiz düzeltme oranını %5'in altında tutmak

Film yorumlarında 84,2 geçilemese bile üç setteki denge ve kural bazında analiz geçerli bir bulgudur.
