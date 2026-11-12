# Görev 01: Veri hazırlığı

**Hafta:** 1. **Tahmini süre:** 10 saat. **GPU:** yok.

## Amaç

Üç test setini ve iki eğitim alt kümesini ortak bir biçimde, sızıntısız ve belgelenmiş olarak hazırlamak.

## Başlamadan önce

`CLAUDE.md`, `docs/01_KARARLAR.md` (özellikle K4, K5, K10, K16) ve `docs/03_TEKNIK_NOTLAR.md` dosyalarını oku.

## Adımlar

### 1. Kaynakları doğrula

Hiçbir sütun adını, Hugging Face kimliğini ya da lisansı varsayma.

- GECTurk: `GGLab/GECTurk` veri setini yükle. Her ayrımın (train, dev, test, movie_reviews) sütunlarını, satır sayısını ve üçer örneğini yazdır.
- GECTurk deposundaki özgün M2 dosyalarını bul (`github.com/GGLAB-KU/gecturk`). Parquet ile M2 arasındaki ilişkiyi anla: kural etiketleri hangisinde?
- OSCAR GEC: `github.com/asimokby/Turkish-GEC` README'sinden eğitim verisinin, test setinin ve mT5 modelinin Hugging Face kimliklerini al. Eğitim verisini `streaming=True` ile aç; tamamını indirme.
- #Turki$hTweets: `github.com/atubakoksal/annotated_tweets` deposunu incele. Biçimi ve etiket şemasını çıkar. Ersoy ve Yıldız'ın değerlendirmede kullandığı sürüm bu depodaki mi, yoksa kendi Hugging Face kopyaları mı? Onların kullandığı sürümü tercih et.
- Her kaynağın lisansını bul.

Bulduklarını `docs/raporlar/01_veri_kaynaklari.md` dosyasına yaz. Belirsiz bir şey varsa (örneğin lisans yoksa) dur ve kullanıcıya sor.

### 2. Tokenizasyonu incele

GECTurk cümlelerinde noktalama ayrı token mı (ör. `geldi .`), yoksa bitişik mi? OSCAR ve tweet setlerinde durum ne? Farklıysa ortak biçimde hangisini koruyacağımızı kullanıcıyla kararlaştır. Kural: skorlayıcının beklediği tokenizasyon bozulmaz.

### 3. Ortak biçime çevir

`scripts/prepare_data.py` içinde her kaynak için bir dönüştürücü yaz. Çıktı `data/processed/<ad>.jsonl`:

```json
{"id": "gecturk-train-000001", "source": "...", "target": "...", "rules": ["CONJ_DE_SEP"], "origin": "gecturk", "split": "train"}
```

- `rules`: GECTurk için kural kimlikleri; hatasız cümlede boş liste. OSCAR ve tweetlerde boş liste (etiket yok ya da farklı şema); özgün etiketler varsa `extra` alanında sakla.
- Özgün dosyalar (M2 dahil) `data/raw/` altında değiştirilmeden kalır; resmi skorlama onları kullanır.
- Her dönüştürücü için küçük bir sahte veriyle test yaz.

### 4. Sızıntı taraması

`trgec.leakage.find_overlap` ile şu çiftleri tara ve sayıları raporla:

| Eğitim tarafı | Test tarafı |
| --- | --- |
| GECTurk train | GECTurk dev, test, film yorumları |
| OSCAR GEC train (kullanılacak alt küme ve doğrulama dilimi) | OSCAR GEC test, film yorumları, tweetler |
| GECTurk train | OSCAR GEC test, tweetler |

Çakışan cümleler eğitim tarafından çıkarılır. Test setleri değiştirilmez.

### 5. Doğrulama dilimi ve alt kümeler

- OSCAR eğitim verisinden 500 cümlelik doğrulama dilimi ayır (`oscar_dev`). Bu dilim eğitim alt kümesine girmez.
- GECTurk train'den `trgec.sampling.stratified_sample` ile 12 bin cümlelik alt küme üret (`clean_ratio=0.5`, `min_per_rule=50`). Bu alt küme D1 ve D2'de aynıdır.
- OSCAR eğitim verisinden 12 bin cümlelik alt küme üret. OSCAR'da hatasız cümle oranını raporla; çok düşükse kullanıcıya sor.
- Tohum `configs/base.yaml` içindeki `seed` değeridir. Seçilen kimlikleri `data/processed/splits/` altına yaz ki alt kümeler yeniden üretilebilsin.

### 6. Veri raporu

`docs/raporlar/01_veri_raporu.md` dosyasına yaz:

- Her set için cümle sayısı, hatalı ve hatasız cümle oranı, ortalama ve en büyük cümle uzunluğu (kelime)
- GECTurk'te kural sıklıkları: tüm eğitim verisinde ve seçilen alt kümede yan yana
- Sızıntı taramasının sonuçları
- Lisans özeti
- Tokenizasyon gözlemi

### 7. Elle kontrol

Her setten 20 rastgele örneği (kaynak, hedef, kurallar) ekrana yazdıran bir komut ekle. Kullanıcı bunları gözle kontrol edecek. Bu adım atlanmaz: biçim hatası burada yakalanmazsa sonraki bütün skorlar bozulur.

## Yapılmayacaklar

- Test setlerinin içeriğine bakarak karar verme (biçim kontrolü dışında).
- Veriyi depoya ekleme.
- Tüm OSCAR eğitim verisini indirme.
- Eğitim verisini küçük harfe çevirme ya da noktalamayı değiştirme.

## Bitti sayılma ölçütleri

- [ ] `python scripts/prepare_data.py --config configs/base.yaml` baştan sona hatasız çalışıyor
- [ ] `data/processed/` altında üç test seti, iki doğrulama seti, iki eğitim alt kümesi var
- [ ] Alt kümede her kuraldan en az 50 örnek var (toplamı 50'den az olanlar hariç)
- [ ] Sızıntı taraması sıfır çakışma bırakıyor
- [ ] İki rapor yazıldı
- [ ] Kullanıcı 20'şer örneği kontrol etti
- [ ] `pytest -q` geçiyor, `docs/GUNLUK.md` güncellendi
