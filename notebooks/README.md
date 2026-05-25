# Notebooks

Thư mục này chứa các notebook phục vụ quá trình khám phá dữ liệu, tái tạo feature, huấn luyện mô hình và so sánh kết quả. Notebook được giữ lại vì đây là đồ án nghiên cứu dữ liệu: cần thể hiện quá trình thử nghiệm, giải thích từng bước và trực quan hóa kết quả.

Khuyến nghị chạy Jupyter từ project root:

```bash
cd /home/dohaidang/DataMining_Project
jupyter lab
```

Các notebook hiện dùng đường dẫn project local trong cell cấu hình. Nếu clone repo sang máy khác, cần chỉnh biến `ROOT` trong notebook hoặc chạy tại đúng cấu trúc thư mục tương ứng.

## Nhóm notebook tạo feature

| Notebook | Dùng để làm gì | Vì sao có file này |
|---|---|---|
| `build_full_60_connectivity_features.ipynb` | Xây dựng bộ 60 functional connectivity feature sets từ dữ liệu epoch và feature có sẵn. | Dùng trong giai đoạn đầu để mở rộng feature theo paper, bảo đảm có đủ tổ hợp `5 bands × 12 metrics`. |
| `recompute_full_60_connectivity_features.ipynb` | Tính lại toàn bộ 60 connectivity feature sets từ `Cleaned_Epochs`. | Dùng khi cần tái tạo đầy đủ feature cache từ dữ liệu EEG đã làm sạch, không phụ thuộc cache cũ. |
| `recompute_15_updated_connectivity_features.ipynb` | Tính lại nhóm connectivity được cập nhật và ghép với các feature còn lại. | Dùng để tiết kiệm thời gian khi chỉ cần cập nhật một phần feature thay vì chạy lại toàn bộ 60 sets. |

## Nhóm notebook huấn luyện và đánh giá mô hình

| Notebook | Dùng để làm gì | Vì sao có file này |
|---|---|---|
| `model_training_losocv_full_60_v4_split.ipynb` | Notebook chính chạy stacked ensemble LOSOCV cho ba bài toán `AD vs HC`, `FTD vs HC`, `FTD vs AD`. | Đây là notebook quan trọng nhất để tạo kết quả cuối theo flow paper: FgMDM base classifiers, Logistic Regression Elastic Net meta-classifier và subject-level LOSOCV. |
| `model_training_losocv_full.ipynb` | Notebook LOSOCV stacked ensemble phiên bản phát triển trước đó. | Giữ lại để thể hiện quá trình phát triển mô hình và đối chiếu với notebook final. |
| `model_training_full_paper_features_baseline.ipynb` | Chạy baseline theo từng feature set connectivity và so sánh các metric/band. | Dùng để đánh giá feature đơn lẻ trước khi dùng stacked ensemble, giúp giải thích feature nào có tín hiệu phân loại tốt. |
| `model_training_precomputed_features.ipynb` | Huấn luyện nhanh từ feature đã tính sẵn. | Dùng trong giai đoạn thử nghiệm để giảm thời gian chạy so với tính feature từ EEG epochs. |
| `model_training_resnet_cnn.ipynb` | Thử nghiệm mô hình ResNet/CNN bổ sung. | Dùng như hướng so sánh deep learning với pipeline functional connectivity + FgMDM. |

## Notebook chính nên đọc trước

Nếu chỉ cần hiểu kết quả cuối của project, đọc theo thứ tự:

1. `model_training_losocv_full_60_v4_split.ipynb`
2. `model_training_full_paper_features_baseline.ipynb`
3. `model_training_resnet_cnn.ipynb`

Nếu cần hiểu cách tạo cache feature, đọc:

1. `recompute_15_updated_connectivity_features.ipynb`
2. `recompute_full_60_connectivity_features.ipynb`

## Output liên quan

Các notebook lưu kết quả vào:

```text
../notebook_outputs/
```

Thư mục output này bị ignore khỏi GitHub vì là artifact sinh ra khi chạy notebook. Các kết quả chính được mô tả lại trong README root và có thể tái tạo nếu chuẩn bị đủ dữ liệu/cache.
