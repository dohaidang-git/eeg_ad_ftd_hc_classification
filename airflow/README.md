# Airflow cho project EEG AD/FTD/HC

## Mục tiêu thiết kế

Thư mục này có hai DAG chính:

| DAG | Mục đích | Có train LOSOCV không? |
|---|---|---:|
| `ftd_eeg_mlflow_final_pipeline` | Kiểm tra cache/kết quả notebook đã chốt, tạo model card và log artifacts sang MLflow. | Không |
| `ftd_eeg_losocv_mlflow_pipeline` | Kiểm tra/tạo connectivity cache, chạy FgMDM + stacked ensemble LOSOCV cho 3 bài toán và log kết quả sang MLflow. | Có |

## DAG final-artifact

DAG `ftd_eeg_mlflow_final_pipeline` dùng để đóng gói pipeline sau khi notebook đã chốt. DAG này không train lại mô hình mặc định, nhưng đã có cơ chế cache-aware cho connectivity features. Nó thực hiện các bước:

1. Kiểm tra feature cache.
2. Nếu cache đủ 60 connectivity feature sets thì dùng lại cache.
3. Nếu cache thiếu thì tính lại connectivity từ `Cleaned_Epochs`.
4. Kiểm tra các file kết quả cuối.
5. Tóm tắt feature cache.
6. Đọc metrics LOSOCV đã sinh từ notebook.
7. Tạo `model_card.md`.
8. Log metrics, report và artifacts vào MLflow.

## DAG LOSOCV training

DAG `ftd_eeg_losocv_mlflow_pipeline` dùng khi muốn Airflow chạy lại phần training LOSOCV từ cache connectivity. Flow:

```text
ensure_feature_cache
        ↓
train_ad_hc_losocv ┐
train_ftd_hc_losocv ├── summarize_losocv_runs
train_ftd_ad_losocv ┘
```

Trong đó:

- Nếu cache đủ 60 feature sets thì dùng lại cache.
- Nếu cache thiếu thì tính lại connectivity từ `Cleaned_Epochs`.
- Mỗi task train gọi `scripts/run_mlflow_experiment.py`.
- Kết quả train được log trực tiếp vào MLflow experiment `ftd_ad_hc_losocv_airflow`.
- Log chạy của từng bài toán được lưu vào `notebook_outputs/airflow_losocv_pipeline/`.

Notebook vẫn được giữ nguyên ở thư mục gốc của project để phục vụ báo cáo, kiểm chứng và chỉnh sửa thủ công. Airflow chỉ là lớp orchestration dùng để kiểm tra, gom kết quả và log MLflow.

## Cấu trúc Airflow

```text
airflow/
├── .env.example
├── Dockerfile
├── README.md
├── docker-compose.yml
├── requirements.txt
├── cache/
│   └── current_features -> ../../Full_MultiDomain_Features_Role3_v5
├── config/
├── logs/
├── plugins/
└── dags/
    └── ftd_eeg_mlflow_final_pipeline.py
```

## Cache dùng cho Airflow

Airflow không đọc trực tiếp bằng tên `Full_MultiDomain_Features_Role3_v5` trong DAG nữa. DAG đọc qua alias ổn định:

```bash
airflow/cache/current_features
```

Alias này đang trỏ tới:

```bash
Full_MultiDomain_Features_Role3_v5
```

Lợi ích: nếu sau này có cache mới, chỉ cần đổi symlink hoặc biến `FTD_EEG_FEATURE_CACHE_DIR`, không cần sửa code DAG.

## Chạy bằng Docker Compose

Từ thư mục project:

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

## Biến môi trường chính

```bash
FTD_EEG_PROJECT_ROOT=/opt/airflow/project
FTD_EEG_FEATURE_CACHE_DIR=/opt/airflow/project/airflow/cache/current_features
FTD_EEG_EPOCH_DIR=/opt/airflow/project/Cleaned_Epochs
FTD_EEG_CONNECTIVITY_WORK_CACHE_DIR=/opt/airflow/project/.cache/connectivity
MLFLOW_TRACKING_URI=file:/opt/airflow/project/mlruns
MLFLOW_EXPERIMENT_NAME=ftd_ad_hc_final_artifacts
MLFLOW_LOSOCV_EXPERIMENT_NAME=ftd_ad_hc_losocv_airflow
FTD_EEG_LOSOCV_OUTER_LIMIT=
```

Nếu muốn test nhanh LOSOCV mà không chạy full, đặt:

```bash
FTD_EEG_LOSOCV_OUTER_LIMIT=4
```

Nếu để rỗng, DAG sẽ chạy full LOSOCV và có thể tốn nhiều thời gian.

## Output của DAG final

DAG tạo thêm folder:

```text
notebook_outputs/airflow_final_pipeline/
├── feature_cache_summary.json
├── final_metrics_payload.json
└── model_card.md
```

## Ghi chú

- DAG này phù hợp để trình bày hướng DA/DE vì nó thể hiện orchestration, validation, artifact tracking và experiment tracking.
- DAG final có thể tự tạo lại connectivity cache nếu thiếu, nhưng không train lại stacked ensemble.
- DAG LOSOCV mới là DAG dùng để train lại FgMDM + Logistic Regression Elastic Net theo LOSOCV và log kết quả sang MLflow.
- Với đồ án hiện tại, nên dùng DAG final khi chỉ cần demo artifact tracking; chỉ chạy DAG LOSOCV khi chấp nhận thời gian train dài.
