# Airflow cho project EEG AD/FTD/HC

## Mục tiêu thiết kế

DAG `ftd_eeg_mlflow_final_pipeline` dùng để đóng gói pipeline sau khi notebook đã chốt. DAG này không train lại mô hình mặc định. Nó thực hiện các bước:

1. Kiểm tra feature cache và các file kết quả cuối.
2. Tóm tắt feature cache.
3. Đọc metrics LOSOCV đã sinh từ notebook.
4. Tạo `model_card.md`.
5. Log metrics, report và artifacts vào MLflow.

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
MLFLOW_TRACKING_URI=file:/opt/airflow/project/mlruns
MLFLOW_EXPERIMENT_NAME=ftd_ad_hc_final_artifacts
```

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
- Nếu muốn train lại thật sự, nên tạo DAG riêng gọi `scripts/run_mlflow_experiment.py`; không nên trộn với DAG final-artifact này để tránh vô tình chạy pipeline nặng.
- Với đồ án hiện tại, DAG final là lựa chọn an toàn vì notebook đã chốt và cache v5 đã có sẵn.
