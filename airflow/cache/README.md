# Airflow feature cache

`current_features` là alias cache đang được Airflow sử dụng.

Hiện tại:

```text
current_features -> ../../Full_MultiDomain_Features_Role3_v5
```

Symlink này không nên commit lên GitHub vì target cache rất lớn và được ignore. Sau khi clone repo và đã có cache cục bộ, tạo lại symlink bằng:

```bash
cd /home/dohaidang/DataMining_Project
ln -s ../../Full_MultiDomain_Features_Role3_v5 airflow/cache/current_features
```

DAG đọc cache qua biến:

```bash
FTD_EEG_FEATURE_CACHE_DIR=/opt/airflow/project/airflow/cache/current_features
```

Nếu cần đổi sang cache khác, cập nhật symlink hoặc đổi biến môi trường trong `.env`; không cần sửa DAG.
