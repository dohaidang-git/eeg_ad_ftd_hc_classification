# EEG-Based AD/FTD/HC Classification with Functional Connectivity

Project này xây dựng pipeline phân loại bệnh nhân Alzheimer disease (AD), frontotemporal dementia (FTD) và nhóm khỏe mạnh (HC) từ tín hiệu EEG. Hướng tiếp cận chính dựa trên functional connectivity, FgMDM, stacked ensemble và đánh giá leave-one-subject-out cross-validation (LOSOCV). Project cũng có phần Airflow + MLflow để đóng gói kết quả cuối theo hướng Data Analytics/Data Engineering/MLOps.

> Lưu ý: project phục vụ mục tiêu học thuật và nghiên cứu dữ liệu. Kết quả không dùng cho chẩn đoán y khoa.

## Bài Toán

Mục tiêu là phân loại EEG theo ba bài toán nhị phân:

| Problem | Positive class | Negative class | Ý nghĩa |
|---|---:|---:|---|
| `ad_hc` | AD | HC | Phân biệt bệnh nhân Alzheimer với người khỏe mạnh |
| `ftd_hc` | FTD | HC | Phân biệt bệnh nhân FTD với người khỏe mạnh |
| `ftd_ad` | FTD | AD | Phân biệt hai nhóm bệnh có biểu hiện gần nhau hơn |

Dataset gốc là EEG BIDS từ OpenNeuro `ds004504`, gồm ba nhóm nhãn:

| Code | Nhóm |
|---|---|
| `A` | Alzheimer disease |
| `F` | Frontotemporal dementia |
| `C` | Healthy control |

## Tóm Tắt Phương Pháp

Pipeline nghiên cứu được triển khai theo các bước:

1. Đọc EEG epochs đã làm sạch từ `Cleaned_Epochs/`.
2. Tách tín hiệu theo năm dải tần: `delta`, `theta`, `alpha`, `beta`, `gamma`.
3. Tính functional connectivity theo 12 thước đo: `cov`, `corr`, `xcov`, `xcorr`, `csd`, `coh`, `mi`, `ecc`, `aecov`, `aecorr`, `plv`, `wplv`.
4. Mỗi cặp `band × metric` tạo thành một feature set, tổng cộng 60 feature sets.
5. Dùng FgMDM làm base classifier cho từng feature set.
6. Tổng hợp xác suất dự đoán từ nhiều base classifiers bằng Logistic Regression Elastic Net.
7. Đánh giá bằng LOSOCV ở mức subject để tránh rò rỉ dữ liệu giữa các epoch của cùng một bệnh nhân.
8. Lưu metrics, subject predictions, fold artifacts, report và MLflow artifacts.

Sơ đồ rút gọn:

```text
Cleaned EEG epochs
        ↓
Band-specific EEG signals
        ↓
60 functional connectivity feature sets
        ↓
FgMDM base classifiers
        ↓
Subject-level probability aggregation
        ↓
Logistic Regression Elastic Net stacked ensemble
        ↓
LOSOCV metrics, plots, report, MLflow artifacts
```

## Kết Quả Chính

Kết quả cuối hiện được lưu tại:

```text
notebook_outputs/full_paper_60_v4_split_all_binary_losocv_metrics_summary.csv
```

Tóm tắt metrics:

| Problem | ROC-AUC | Accuracy | F1 | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| AD vs HC | 0.8276 | 0.7077 | 0.7532 | 0.8056 | 0.5862 |
| FTD vs HC | 0.7616 | 0.6923 | 0.6667 | 0.6957 | 0.6897 |
| FTD vs AD | 0.6292 | 0.6102 | 0.3784 | 0.3043 | 0.8056 |
| Mean | 0.7395 | 0.6701 | 0.5994 | 0.6019 | 0.6938 |

Nhận xét nhanh:

- `AD vs HC` là bài toán tốt nhất trong ba bài toán, với ROC-AUC cao hơn 0.82.
- `FTD vs HC` đạt mức trung bình khá, cân bằng hơn giữa sensitivity và specificity.
- `FTD vs AD` là bài toán khó nhất vì hai nhóm đều là bệnh lý thần kinh, tín hiệu EEG có mức chồng lấn cao hơn; sensitivity thấp cho thấy mô hình còn bỏ sót nhiều subject FTD khi so với AD.

## Cấu Trúc Project

```text
DataMining_Project/
├── README.md
├── requirements.txt
├── RUN_MLFLOW.md
├── s41598-026-35316-9.pdf                  # optional local paper PDF, ignored
├── src/
│   └── ftd_mlflow_pipeline/
│       ├── connectivity.py
│       ├── experiment.py
│       └── precomputed_features.py
├── scripts/
│   └── run_mlflow_experiment.py
├── airflow/
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── cache/
│   │   └── current_features -> ../../Full_MultiDomain_Features_Role3_v5
│   └── dags/
│       └── ftd_eeg_mlflow_final_pipeline.py
├── notebooks/
│   ├── README.md
│   ├── build_full_60_connectivity_features.ipynb
│   ├── model_training_precomputed_features.ipynb
│   ├── model_training_losocv_full.ipynb
│   ├── model_training_losocv_full_60_v4_split.ipynb
│   ├── model_training_full_paper_features_baseline.ipynb
│   ├── model_training_resnet_cnn.ipynb
│   ├── recompute_15_updated_connectivity_features.ipynb
│   └── recompute_full_60_connectivity_features.ipynb
├── Cleaned_Epochs/                         # ignored: large local data
├── ds004504/                               # ignored: raw OpenNeuro dataset
├── Full_MultiDomain_Features_Role3_v5/     # ignored: feature cache for Airflow
├── mlruns/                                 # ignored: local MLflow store
└── notebook_outputs/                       # ignored: generated outputs
```

Các thư mục dữ liệu lớn được ignore khỏi GitHub. Người dùng cần tải hoặc tạo lại dữ liệu cục bộ trước khi chạy toàn bộ project.

## Cài Đặt Môi Trường Notebook

Khuyến nghị dùng Python 3.10 hoặc 3.11.

```bash
cd /home/dohaidang/DataMining_Project
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name ftd-eeg --display-name "FTD EEG"
```

Nếu dùng Jupyter:

```bash
jupyter lab
```

Sau đó chọn kernel `FTD EEG`.

## Chuẩn Bị Dữ Liệu

Project hiện giả định các thư mục sau tồn tại cục bộ:

```text
ds004504/
Cleaned_Epochs/
Full_MultiDomain_Features_Role3_v5/
```

Trong đó:

- `ds004504/`: dataset OpenNeuro gốc hoặc bản tải về bằng DataLad/OpenNeuro.
- `Cleaned_Epochs/`: EEG epochs đã làm sạch, tách theo subject, label và band.
- `Full_MultiDomain_Features_Role3_v5/`: cache 60 connectivity feature sets, dùng cho notebook và Airflow để tránh tính lại từ đầu.

Kiểm tra nhanh cache:

```bash
ls Full_MultiDomain_Features_Role3_v5/labels.npy
ls Full_MultiDomain_Features_Role3_v5/subject_ids.npy
ls Full_MultiDomain_Features_Role3_v5/feature_metadata.csv
```

## Chạy Notebook

Các notebook chính:

| Notebook | Mục đích |
|---|---|
| `notebooks/model_training_losocv_full_60_v4_split.ipynb` | Notebook chính cho stacked ensemble LOSOCV trên ba bài toán nhị phân |
| `notebooks/model_training_full_paper_features_baseline.ipynb` | Baseline theo từng feature set connectivity |
| `notebooks/model_training_resnet_cnn.ipynb` | Mô hình bổ sung ResNet/CNN |
| `notebooks/recompute_15_updated_connectivity_features.ipynb` | Tính lại một phần connectivity và ghép cache |
| `notebooks/recompute_full_60_connectivity_features.ipynb` | Tính lại toàn bộ 60 connectivity feature sets |

Xem thêm `notebooks/README.md` để biết vai trò và lý do tồn tại của từng notebook.

Nếu chỉ muốn xem kết quả đã chốt, mở:

```text
notebooks/model_training_losocv_full_60_v4_split.ipynb
```

Nếu muốn chạy lại toàn bộ từ feature cache, kiểm tra biến đường dẫn trong notebook trỏ về:

```python
PRECOMPUTED_DIR = Path("Full_MultiDomain_Features_Role3_v5")
```

Output chính sau khi chạy:

```text
notebook_outputs/full_paper_60_v4_split_ad_hc_losocv_metrics.csv
notebook_outputs/full_paper_60_v4_split_ftd_hc_losocv_metrics.csv
notebook_outputs/full_paper_60_v4_split_ftd_ad_losocv_metrics.csv
notebook_outputs/full_paper_60_v4_split_all_binary_losocv_metrics_summary.csv
notebook_outputs/full_paper_60_v4_split_paper_comparison_metrics.csv
```

## Chạy MLflow Bằng Script

Script MLflow nằm ở:

```text
scripts/run_mlflow_experiment.py
```

Ví dụ smoke test:

```bash
cd /home/dohaidang/DataMining_Project
python scripts/run_mlflow_experiment.py \
  --problem ad_hc \
  --bands alpha \
  --metrics cov,corr \
  --inner-folds 3 \
  --outer-limit 4
```

Chạy với precomputed features:

```bash
python scripts/run_mlflow_experiment.py \
  --feature-source from_precomputed \
  --precomputed-dir Full_MultiDomain_Features_Role3_v5 \
  --problem all \
  --bands delta,theta,alpha,beta,gamma \
  --metrics cov,corr,xcov,xcorr,csd,coh,mi,ecc,aecov,aecorr,plv,wplv
```

Mở MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Sau đó truy cập:

```text
http://localhost:5000
```

## Chạy Airflow Và MLflow UI

Airflow được đặt trong thư mục riêng:

```text
airflow/
```

DAG hiện tại:

```text
airflow/dags/ftd_eeg_mlflow_final_pipeline.py
```

DAG này không train lại notebook. Nó dùng kết quả đã chốt và cache `Full_MultiDomain_Features_Role3_v5` để:

1. Validate feature cache và output files.
2. Tóm tắt feature cache.
3. Đọc metrics cuối.
4. Sinh `model_card.md`.
5. Log metrics và artifacts vào MLflow.

Cache cho Airflow được trỏ qua symlink:

```text
airflow/cache/current_features -> ../../Full_MultiDomain_Features_Role3_v5
```

Nếu clone repo mới, tạo lại symlink này sau khi đã có cache cục bộ:

```bash
cd /home/dohaidang/DataMining_Project
ln -s ../../Full_MultiDomain_Features_Role3_v5 airflow/cache/current_features
```

Điều này giúp DAG luôn đọc qua một đường dẫn ổn định:

```bash
FTD_EEG_FEATURE_CACHE_DIR=/opt/airflow/project/airflow/cache/current_features
```

### Chạy Airflow bằng Docker Compose

```bash
cd /home/dohaidang/DataMining_Project/airflow
cp .env.example .env
docker compose up airflow-init
docker compose up -d airflow-webserver airflow-scheduler mlflow-ui
```

Mở giao diện:

```text
Airflow: http://localhost:8080
MLflow:  http://localhost:5000
```

Tài khoản Airflow mặc định:

```text
username: airflow
password: airflow
```

Trong Airflow UI:

1. Tìm DAG `ftd_eeg_mlflow_final_pipeline`.
2. Unpause DAG.
3. Bấm Trigger DAG.
4. Kiểm tra task logs.
5. Mở MLflow UI để xem run `final_notebook_outputs_logged_by_airflow`.

Output của DAG:

```text
notebook_outputs/airflow_final_pipeline/
├── feature_cache_summary.json
├── final_metrics_payload.json
└── model_card.md
```

### Dừng Airflow

```bash
cd /home/dohaidang/DataMining_Project/airflow
docker compose down
```

Nếu muốn xóa database metadata Airflow local:

```bash
docker compose down -v
```

## Chính Sách Dữ Liệu Khi Đưa Lên GitHub

Không nên commit các thư mục/file sau:

```text
ds004504/
Cleaned_Epochs/
Full_MultiDomain_Features_Role3/
Full_MultiDomain_Features_Role3_v5/
Full_MultiDomain_Features_Role3_v5.tar.zst
.cache/
mlruns/
notebook_outputs/
```

Lý do:

- Dung lượng rất lớn, không phù hợp GitHub thường.
- Dataset gốc nên được tải từ nguồn chính thức.
- Feature cache và MLflow outputs là artifact sinh ra, không phải source code.

Nếu cần chia sẻ artifact, nên dùng một trong các cách:

- GitHub Releases cho file nén nhỏ.
- Google Drive/OneDrive cho cache lớn.
- DVC, Git LFS hoặc object storage nếu muốn quản lý dữ liệu nghiêm túc.

## Tài Liệu Tham Khảo

1. T. Mlinarič, A. Van Den Kerchove, Z. I. Barinaga, and M. M. Van Hulle, "EEG-based classification of Alzheimer's disease and frontotemporal dementia using functional connectivity," Scientific Reports, 2026.
2. A. Miltiadous et al., "A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and healthy subjects from routine EEG," Data, 2023.
3. A. Barachant, S. Bonnet, M. Congedo, and C. Jutten, "Riemannian geometry applied to BCI classification," 2010.
4. D. H. Wolpert, "Stacked generalization," Neural Networks, 1992.
5. H. Zou and T. Hastie, "Regularization and variable selection via the elastic net," Journal of the Royal Statistical Society Series B, 2005.
