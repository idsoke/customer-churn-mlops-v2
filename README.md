# Customer Churn Predictor

Aplikasi untuk mendeteksi risiko pelanggan e-commerce berhenti (*churn*) sebelum mereka benar-benar pergi. Proyek ini mencakup seluruh alur MLOps: eksplorasi data, pipeline training yang bisa direproduksi dengan DVC, API prediksi FastAPI, dan antarmuka web Angular.

```
┌──────────────────────┐   POST /predict    ┌──────────────────────────┐
│  Frontend (Angular)  │ ─────────────────▶ │   Backend API (FastAPI)  │
│  form data pelanggan │ ◀───────────────── │  RandomForest + threshold│
│  + hasil & performa  │   GET /model/info  └────────────▲─────────────┘
└──────────────────────┘                                 │ models/*.pkl, *.json
                                                         │
                         ┌───────────────────────────────┴──────────┐
                         │  Pipeline DVC: preprocess ──▶ train      │
                         │  data/raw/*.xlsx ──▶ data/processed/*.csv │
                         └──────────────────────────────────────────┘
```

## Performa model

Hasil evaluasi pada test set (20% data, stratified), dari `backend/metrics.json`:

| Metrik    | Nilai |
| --------- | ----- |
| Accuracy  | 0.960 |
| Precision | 0.914 |
| Recall    | 0.842 |
| F1        | 0.877 |
| ROC-AUC   | 0.988 |

Model memakai `RandomForestClassifier` (200 pohon, `max_depth` 10, `class_weight=balanced`). Data target tidak seimbang (sekitar 17% churn), jadi threshold tidak memakai 0.5. Threshold dipilih dari precision-recall curve dengan memaksimalkan F1, yaitu **0.635**. Nilai ini disimpan di `models/threshold.json` dan dipakai juga oleh API, supaya hasil prediksi konsisten dengan metrik di atas.

## Struktur proyek

```
.
├── backend/
│   ├── api/                 # FastAPI: main.py (endpoint), schemas.py, inference.py
│   ├── src/                 # Pipeline: config.py, preprocess.py, train.py
│   ├── data/
│   │   ├── raw/             # Dataset mentah (dilacak DVC, tidak masuk Git)
│   │   └── processed/       # train.csv & test.csv hasil stage preprocess
│   ├── models/              # churn_model.pkl, feature_columns.json, threshold.json
│   ├── notebooks/           # exploratory_analysis.ipynb (EDA)
│   ├── tests/               # Unit test & end-to-end test pipeline (pytest)
│   ├── dvc.yaml / dvc.lock  # Definisi & lock pipeline
│   ├── params.yaml          # Hyperparameter preprocess & train
│   ├── metrics.json         # Metrik model terakhir
│   └── Dockerfile           # Image untuk serving API (port 7860)
└── frontend/                # Angular 22: form prediksi + kartu performa model
```

## Backend

Butuh Python 3.12 atau lebih baru.

### Menjalankan API

Model hasil training (`churn_model.pkl`, `feature_columns.json`, `threshold.json`) sudah ikut di repo, jadi API bisa langsung dijalankan tanpa training ulang.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .            # dependensi API dari pyproject.toml
cp .env.example .env        # opsional

fastapi dev                 # http://localhost:8000, dokumentasi di /docs
```

### Endpoint

| Method | Path          | Keterangan                                       |
| ------ | ------------- | ------------------------------------------------ |
| GET    | `/health`     | Cek API hidup                                    |
| GET    | `/model/info` | Metrik model (accuracy, precision, recall, F1, ROC-AUC) |
| POST   | `/predict`    | Prediksi churn untuk satu pelanggan              |

Contoh request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Tenure": 12, "PreferredLoginDevice": "Mobile Phone", "CityTier": 1,
    "WarehouseToHome": 15, "PreferredPaymentMode": "Credit Card", "Gender": "Male",
    "HourSpendOnApp": 3.5, "NumberOfDeviceRegistered": 4,
    "PreferedOrderCat": "Laptop & Accessory", "SatisfactionScore": 3,
    "MaritalStatus": "Single", "NumberOfAddress": 2, "Complain": 0,
    "OrderAmountHikeFromlastYear": 15, "CouponUsed": 1, "OrderCount": 2,
    "DaySinceLastOrder": 5, "CashbackAmount": 150.5
  }'
```

Contoh response:

```json
{ "churn": false, "churn_probability": 0.1234, "risk_level": "Aman" }
```

`risk_level` bernilai `"Tinggi"` jika probabilitas ≥ threshold, selain itu `"Aman"`. Input dikirim dalam bentuk mentah, sama seperti kolom di dataset asli. API yang melakukan one-hot encoding sesuai urutan kolom saat training (`feature_columns.json`). Daftar nilai yang valid untuk setiap field ada di `api/schemas.py`.

### Konfigurasi

| Variabel          | Default                 | Keterangan                                   |
| ----------------- | ----------------------- | -------------------------------------------- |
| `ALLOWED_ORIGINS` | `http://localhost:4200` | Origin frontend untuk CORS, pisahkan dengan koma |
| `DATA_DIR`        | `data/raw`              | Folder dataset mentah                        |
| `MODEL_DIR`       | `models`                | Folder artefak model                         |
| `MODEL_NAME`      | `churn_model.pkl`       | Nama file model                              |

### Training ulang (pipeline DVC)

Dataset mentah tidak ikut di Git. Letakkan `E Commerce Dataset.xlsx` (sheet `E Comm`) di `backend/data/raw/`, lalu:

```bash
pip install dvc pyyaml openpyxl pytest
dvc init --subdir           # sekali saja; folder .dvc belum ada di repo
dvc repro                   # jalankan stage yang berubah saja
dvc metrics show            # lihat metrik terbaru
```

Pipeline terdiri dari dua stage:

1. **preprocess** (`src/preprocess.py`) membuang kolom `CustomerID`, mengisi nilai kosong pada kolom numerik dengan median, lalu menggabungkan kategori yang sama tapi ditulis berbeda (`CC` menjadi `Credit Card`, `COD` menjadi `Cash on Delivery`, `Mobile` menjadi `Mobile Phone`). Setelah itu kolom kategorikal di-*one-hot encode* dan data di-split train/test secara stratified.
2. **train** (`src/train.py`) melatih RandomForest, mencari threshold terbaik, lalu menyimpan model, daftar kolom fitur, threshold, dan `metrics.json`.

Hyperparameter diatur di `params.yaml`. DVC otomatis menjalankan ulang stage yang parameternya berubah.

### Test

```bash
cd backend
pytest tests/
```

Test mencakup fungsi preprocessing (imputasi, encoding, normalisasi kategori), pemisahan fitur dan target, serta pipeline end-to-end dari preprocess sampai train.

### Docker

```bash
cd backend
docker build -t churn-api .
docker run -p 7860:7860 churn-api
```

> Catatan: `Dockerfile` membaca `requirements-api.txt`, yang belum ada di repo ini. Buat dulu file tersebut dari dependensi di `pyproject.toml` sebelum build.

## Frontend

Butuh Node.js dan npm.

```bash
cd frontend
npm install
npm start                   # http://localhost:4200
```

Frontend memanggil API di `http://localhost:8000` saat development. Untuk production, URL API diambil dari `src/environments/environment.ts`. Halaman berisi form data pelanggan (dengan nilai default yang bisa di-reset), hasil prediksi beserta tingkat risikonya, dan kartu performa model dari `/model/info`.

```bash
npm run build               # output: dist/fe-customer-churn-mlops/browser
npm test                    # unit test (Vitest)
```

Deploy ke Vercel sudah dikonfigurasi lewat `frontend/vercel.json`. Jangan lupa menambahkan domain frontend ke `ALLOWED_ORIGINS` di backend.

## Tech stack

- **ML**: pandas, scikit-learn, joblib
- **MLOps**: DVC (pipeline, params, metrics), pytest
- **API**: FastAPI, Pydantic, Uvicorn
- **Frontend**: Angular 22 (standalone component, signals, reactive forms), Vitest

Nilai churn_probability di contoh response hanya ilustrasi. Hasil sebenarnya tergantung model.
