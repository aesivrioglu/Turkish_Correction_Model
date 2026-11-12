# Görev 02: Değerlendirme hattı ve temel çizgiler

**Hafta:** 2. **Tahmini süre:** 10 saat. **GPU:** yaklaşık 1 saat.

## Amaç

Bir tahmin dosyasını alıp her test setinde resmi araçla P, R, F0.5 ve güven aralığı üreten hattı kurmak ve yayımlanmış bir skoru yeniden üreterek doğrulamak.

## Başlamadan önce

`docs/01_KARARLAR.md` içinde K15, K17, K18, K19 maddelerini oku.

## Adımlar

### 1. Hangi skorlayıcı?

Ersoy ve Yıldız'ın makalesini (arXiv 2405.15320) ve deposunu oku: OSCAR test ve tweet skorları hangi araçla hesaplanmış? Bulguyu `docs/raporlar/02_skorlayici.md` dosyasına yaz. Belirsizse kullanıcıya sor; tahminle ilerleme.

**2026-11-13:** Araç belli: ERRANT-TR (K15 notu, `raporlar/00_varsayim_dogrulama.md`). Bu adımda kalan iş, `compare_m2.py` komutunun bayraklarını (`-b 0.5`, `-cse`, `-cat` vb.) belirlemek ve `02_skorlayici.md` dosyasına yazmak. Yazarların deposunda betik yok; bayraklar Kapı A'da 79,0 üretilerek doğrulanır.

### 2. mT5'i yeniden üret (Kapı A)

- OSCAR GEC ile eğitilmiş mT5 modelini (`asimokby/Turkish-OSCAR-GEC-v0`) üç test setinde çalıştır (`scripts/predict.py` içinde genel bir seq2seq yolu ya da ayrı küçük bir betik). GPU Kaggle'da; önce yerelde 5 cümleyle dene.
- Tahminleri JSONL olarak kaydet.
- Tweet setine Ersoy ve Yıldız'ın iki dönüşümünü uygula: altın cümlelerin ilk harfi büyütülür, model çıktılarındaki noktalama silinir.
- Tweet setinde ERRANT-TR ile F0.5 hesapla. Hedef: 79,0 ± 2. İkincil kontrol: OSCAR testinde 55,4, film yorumlarında 28,1 (ikisi de ERRANT-TR).

Tutmuyorsa sırayla kontrol et: tokenizasyon, büyük ve küçük harf, test setinin sürümü, üretim ayarları (ışın sayısı, en fazla uzunluk), skorlayıcı sürümü.

### 3. M2 scorer

- GECTurk deposundaki M2 scorer'ı film yorumları için kur.
- Cümle başına zaman aşımı ekle (`configs/base.yaml` içinde `m2_timeout_seconds`).
- Altın düzeltmeyi hipotez olarak verince F0.5 = 1,0; kaynak cümleyi hipotez olarak verince recall = 0 çıktığını test et.

### 4. Etiketleyici (en fazla yarım gün)

`GGLab/gec-tr-seq-tagger` modelini film yorumlarında çalıştırmayı dene. Hedef: 84,2 ± 2. Yarım günde çalışmazsa bırak, `docs/GUNLUK.md` dosyasına nedenini yaz; K17 uyarınca yayımlanmış skor referans alınır.

### 5. Cümle bazlı sayılar ve bootstrap

Skorlayıcıdan cümle başına (tp, fp, fn) sayılarını çıkar ve `trgec.bootstrap` işlevlerine bağla. Toplam sayılardan hesaplanan F0.5 ile skorlayıcının kendi F0.5 çıktısının aynı olduğunu test et.

### 6. ERRANT-TR (2. adımdan önce kurulur)

Yayımlanmış ve kullanılabilir olduğu doğrulandı (`github.com/harunuz/erranttr`, MIT). Kur:

- Bağımlılıklar `rapidfuzz` ve `zemberek-python`. Eklemeden önce kullanıcıya sor.
- İki kaynak dosya (`tr_TR.aff`, `TS_Corpus_Turkish_Word_List.txt`) deponun 0.1.0 sürüm sayfasından `tr/resources/` altına konur.
- Depo paketlenmemiş ve `process.py` sabit bir dosya adı okuyor. İnce bir sarmalayıcı yaz: paralel cümleler → M2 → `compare_m2.py`.
- M2 scorer'daki akıl sağlığı testlerinin aynısını uygula: altın düzeltme hipotez olarak verilince F0.5 = 1,0; kaynak cümle verilince recall = 0.

ERRANT-TR hem skorlayıcı (OSCAR, tweetler, film yorumları) hem de hata türü analizi için kullanılır (K15).

### 7. `scripts/evaluate.py`

Tek komutla: tahmin dosyası + test seti adı → P, R, F0.5, %95 güven aralığı, aşırı düzeltme oranı (hatasız cümlelerde), bozuk çıktı oranı. Çıktı `outputs/eval/` altına JSON olarak yazılır.

## Kapı A

mT5'in tweet skoru ±2 puan içinde yeniden üretildi. Bu sağlanmadan görev 03'teki Kaggle eğitimleri çalıştırılmaz. (Görev 03'ün yerel kısmı paralel ilerleyebilir.)

## Bitti sayılma ölçütleri

- [ ] Skorlayıcı raporu yazıldı
- [ ] Kapı A geçildi; elde edilen skor `docs/GUNLUK.md` içinde
- [ ] M2 scorer akıl sağlığı testleri geçiyor
- [ ] ERRANT-TR kuruldu, akıl sağlığı testleri geçiyor
- [ ] Etiketleyici denemesinin sonucu yazıldı
- [ ] `scripts/evaluate.py` üç test seti ve iki doğrulama seti için çalışıyor
- [ ] `pytest -q` geçiyor
