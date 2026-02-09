# Azerbaijani YouTube Dataset + Word2Vec/FastText + GRU (Domain Analysis)

Bu repo, Part-2 ödevi için uçtan uca pipeline iskeletini içerir:
- Part-1 etiketli (labeled) veri ile **3-sınıflı** (neg/neu/pos) sentiment eğitimi ve **Macro-F1** değerlendirmesi
- YouTube'dan **etiketsiz** (unlabeled) Azerbaycanca yorum toplama (5 domain)
- **Azerbaycan Türkçesi-only filtre** (Türkçe karışmasın)
- Her video için **tek Excel** export (A1 link + domain/comment satırları)
- Word2Vec/FastText embedding (combined corpus) + GRU (frozen vs fine-tuned) 4 koşu
- Part-1 test datası üzerinde domain-wise true performance + domain shift analizi

## 0) Kurulum

### Python ortamı
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### YouTube API Key (zorunlu)
YouTube Data API v3 anahtarını bir `.env` dosyasında saklayın:

```bash
cp .env.example .env
# .env içine YOUTUBE_API_KEY=... yazın
```

## 1) Proje yapısı (özet)
- `deliverables/` : Teslim edilecek 5 domain klasörü ve video başına Excel dosyaları
- `data/` : Part-1 ve YouTube ara çıktılar
- `models/` : embedding + GRU model çıktıları
- `code/` : pipeline scriptleri (numaralı çalıştırma sırası)
- `report/` : rapor dosyaları/tablolar/figürler

## 2) Çalıştırma sırası (önerilen)
1. **Part-1 master dataset** üret:
```bash
python code/01_part1_prepare_master.py
```

2. Video keşfi + metadata çek:
```bash
python code/02_youtube_discover_videos.py --max_videos_per_domain 60
```

3. Metadata'dan domain atama (rule-based scoring):
```bash
python code/03_domain_assign_metadata.py
```

4. Yorum çekme (pagination):
```bash
python code/04_youtube_fetch_comments.py --max_comments_per_video 5000
```

5. AZ-only filtre + Excel export (deliverables klasörünü doldurur):
```bash
python code/05_filter_comments_and_export_excel.py --threshold 2.5
```

6. Corpus oluştur + Word2Vec/FastText eğit (combined):
```bash
python code/06_build_corpora_and_train_embeddings.py
```

7. GRU 4 koşu (W2V/FT x frozen/tuned) + overall Macro-F1:
```bash
python code/07_train_gru_4runs.py
```

8. Domain-wise true performance (Experiment A):
```bash
python code/08_evaluate_domainwise_A.py
```

9. Domain shift (Experiment B: LODO varsayılan):
```bash
python code/09_evaluate_domainshift_B.py --mode lodo
```

10. YouTube üzerinde betimleyici tahmin dağılımları (performans değil):
```bash
python code/10_predict_youtube_domain_distributions.py
```

## 3) Önemli notlar
- YouTube verisi **etiketsizdir**: YouTube üzerinde accuracy/F1 raporlanmaz.
- Excel export **strict**: A1 video linki; 2. satırdan itibaren A=domain, B=comment. Username vb. saklanmaz.
- Domain isimleri birebir:
  - Technology & Digital Services
  - Finance & Business
  - Social Life & Entertainment
  - Retail & Lifestyle
  - Public Services

## 4) Konfigürasyon
Tüm path/parametreler `code/config.py` içindedir.
