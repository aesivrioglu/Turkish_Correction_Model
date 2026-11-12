# Görev 04 ile 06: Taslak

Bu görevlerin ayrıntılı tanımı, önceki görevin sonuçlarına göre yazılacak. Aşağıdakiler yalnızca çerçevedir.

## Görev 04: İki eğitim (H4 ile H5)

- D1 ve D2'yi Kapı B'de dondurulan ayarlarla eğit.
- Emniyet kemeri eşiğini GECTurk dev ve OSCAR dev üzerinde seç: F0.5'i en çok artıran ya da aşırı düzeltmeyi %5'in altına indiren en gevşek eşik. Test setlerine bakma.
- İki modelin üç test setindeki tahminlerini kemerli ve kemersiz üret.
- Sonuç tablosu: 4 sistem (etiketleyici, mT5, D1, D2) × 3 test seti; P, R, F0.5 ve güven aralığı.

## Görev 05: Analiz ve kural kartları (H6)

- Film yorumlarında kural bazında recall: D1 ve D2 hangi kurallarda etiketleyicinin gerisinde?
- OSCAR ve tweetlerde hata türü bazında analiz (ERRANT-TR ya da veri setinin kendi etiketleri).
- D1 ile D2 arasında eşleştirilmiş bootstrap (AS2'nin cevabı).
- Aşırı düzeltme ve bozuk çıktı oranları.
- 25 kural kartı: `data/kural_kartlari.yaml`. Her kart: kural kimliği, başlık, bir cümlelik kural, bir doğru ve bir yanlış örnek, TDK bağlantısı. Taslağı yapay zeka yazar, kullanıcı her kartı TDK sayfasıyla karşılaştırır.
- 15 ile 20 başarısız örneğin incelemesi.

## Görev 06: Demo, rapor, sunum (H7 ile H8)

- Gradio demosu: metin kutusu, cümlelere bölme, hataların altını çizme, tıklayınca kural kartı. Hugging Face Spaces.
- Ücretsiz katman CPU olduğu için gecikmeyi ölç; gerekirse adaptörü birleştirip 4-bit GGUF'a çevir.
- Rapor ve figürler `docs/rapor/` altında.
- Sunum ve canlı demo provası; demo çökerse gösterilecek ekran kaydı.
