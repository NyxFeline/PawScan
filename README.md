# 🐾 PawScan — Nhận Diện Chó / Mèo

Ứng dụng web nhận diện ảnh Chó/Mèo, xây dựng cho đồ án cuối kỳ môn Machine Learning. Người dùng tải ảnh lên, hệ thống trích xuất đặc trưng và trả về kết quả dự đoán (Chó / Mèo / Không xác định) kèm ảnh đối chiếu.

## Công nghệ sử dụng

**Huấn luyện mô hình (Jupyter Notebook)**

- `OpenCV (cv2)` — đọc ảnh, resize giữ tỉ lệ (resize + padding)
- `scikit-image (skimage.feature.hog)` — trích xuất đặc trưng HOG (Histogram of Oriented Gradients) từ ảnh grayscale
- `scikit-learn` — pipeline `StandardScaler → PCA → LogisticRegression`, tinh chỉnh siêu tham số bằng `GridSearchCV`
- `joblib` — lưu (serialize) pipeline đã huấn luyện ra file `.pkl`

**Backend**

- `Flask` — server xử lý upload ảnh, chạy tiền xử lý + dự đoán, render giao diện
- `joblib.load()` — nạp lại pipeline đã huấn luyện (không dùng `pickle` trực tiếp vì joblib có định dạng lưu mảng numpy riêng)

**Frontend**

- HTML template (Jinja2) + CSS thuần, không dùng framework ngoài
- JavaScript thuần: xem trước ảnh trước khi submit, nút xóa ảnh, trạng thái loading khi đang dự đoán
- Font: `Baloo 2` (tiêu đề), `Be Vietnam Pro` (nội dung), `JetBrains Mono` (nhãn nhỏ) — load qua Google Fonts

## Luồng hoạt động

```
Người dùng chọn ảnh
        │
        ▼
JS đọc file, hiển thị preview ngay (chưa gửi server)
        │
        ▼
Bấm "Dự đoán ngay" → form POST ảnh lên Flask (/)
        │
        ▼
Flask: resize + pad ảnh → grayscale → trích xuất đặc trưng HOG
        │
        ▼
Pipeline (StandardScaler → PCA → LogisticRegression).predict_proba()
        │
        ▼
So sánh xác suất với ngưỡng tin cậy (0.75)
   ├─ Chó ≥ 0.75  → "Đây là Chó"
   ├─ Mèo ≥ 0.75  → "Đây là Mèo"
   └─ còn lại     → "Không xác định"
        │
        ▼
Render lại trang: ảnh gốc + huy hiệu kết quả đặt cạnh nhau
```

**Lưu ý quan trọng:** bước tiền xử lý ảnh ở `app.py` (resize + pad về 128×128, chuyển grayscale, trích HOG với `orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)`) phải khớp **chính xác** với bước tiền xử lý lúc huấn luyện trong notebook — nếu lệch, mô hình sẽ dự đoán sai dù không báo lỗi.

## Cấu trúc thư mục

```
├── app.py                                    # Flask server: load model, tiền xử lý, dự đoán
├── pipeline_catdog_model_final.pkl           # Pipeline đã huấn luyện (joblib)
├── requirements.txt                          # Thư viện Python cần cài
├── dockerfile                                # Đóng gói app thành container
├── .dockerignore                             # Loại trừ file không cần thiết khi build image
├── .gitignore                                # Loại trừ file không cần thiết khi push GitHub
├── templates/
│   └── index.html                            # Giao diện upload + hiển thị kết quả
├── CatDogLogisticRegression_Optimized.ipynb  # Notebook huấn luyện mô hình
└── Cat_Dog_data/                              # Dataset ảnh gốc dùng để train/test (không push lên GitHub)
    ├── train/
    │   ├── cat/
    │   └── dog/
    └── test/
        ├── cat/
        └── dog/
```

> ⚠️ File Docker của project hiện đặt tên là `dockerfile` (chữ thường). Trên Windows điều này không sao vì hệ điều hành không phân biệt hoa/thường, nhưng khi build trên môi trường Linux (như Render) mặc định chỉ tự nhận diện file tên chính xác là `Dockerfile` (chữ D hoa). Nên đổi tên thành `Dockerfile` trước khi push lên GitHub để tránh lỗi "không tìm thấy Dockerfile" lúc deploy.
>
> `Cat_Dog_data/` chỉ dùng để train/test trong notebook, không cần thiết cho việc chạy web app hay build Docker image — đã được loại trừ trong `.gitignore` và `.dockerignore`.

## Cách chạy

**requirements.txt** hiện tại:

```
Flask==2.3.2
scikit-learn
scikit-image
opencv-python-headless==4.8.0.76
numpy>=1.26.4,<2
joblib
gunicorn==21.2.0
```

> `numpy` bị giới hạn trần `<2` vì `opencv-python-headless==4.8.0.76` được build cho NumPy 1.x — nếu để NumPy tự do lên bản 2.x, `import cv2` sẽ lỗi `_ARRAY_API not found` (đặc biệt dễ gặp khi build Docker trên môi trường sạch).

### Chạy trực tiếp (không Docker)

```bash
# 1. Cài thư viện từ requirements.txt
pip install -r requirements.txt

# 2. Đảm bảo pipeline_catdog_model_final.pkl nằm cùng thư mục với app.py

# 3a. Chạy dev server
python app.py

# 3b. Hoặc chạy bằng gunicorn (phù hợp môi trường production/deploy)
gunicorn -b 0.0.0.0:5000 app:app
```

Mặc định server chạy tại `http://localhost:5000`.

### Chạy bằng Docker

**Bước 0 — Cài Docker Desktop (nếu máy chưa có):**

- Tải tại [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) (Windows cần bật WSL2, cài đặt sẽ tự nhắc nếu thiếu).
- Cài xong, mở Docker Desktop lên (để nó chạy nền), rồi kiểm tra bằng terminal:
  ```bash
  docker --version
  ```

**Bước 1 — Build image** (chạy tại thư mục chứa `Dockerfile`):

```bash
docker build -t pawscan .
```

**Bước 2 — Chạy container ở local để test trước khi deploy:**

```bash
docker run -p 5000:5000 pawscan
```

Mở `http://localhost:5000` — nếu chạy được y hệt lúc `python app.py` thì image đã sẵn sàng để deploy.

**Lỗi thường gặp:**

- `docker: command not found` → chưa cài Docker Desktop hoặc chưa mở app lên.
- Build lỗi vì thiếu `pipeline_catdog_model_final.pkl` → file model phải nằm cùng cấp với `Dockerfile` trước khi build (đã khai báo `COPY` trong Dockerfile, Docker sẽ không tự tải file này).
- Container chạy nhưng vào `localhost:5000` không được → kiểm tra đã map đúng cổng `-p 5000:5000` chưa.

### Deploy lên Render (để lấy URL nộp bài)

1. Push toàn bộ project lên một GitHub repo.
2. Vào [render.com](https://render.com) → đăng ký → **New → Web Service** → connect GitHub repo vừa push.
3. Render tự phát hiện `Dockerfile` → chọn Environment = **Docker**.
4. Chọn Instance Type = **Free** → bấm **Deploy**.
5. Đợi build xong, Render cấp 1 URL dạng `https://pawscan.onrender.com` (đặt tên service là `pawscan` lúc tạo) — đây là link nộp bài.

> ⚠️ Free tier của Render sẽ **sleep sau 15 phút không có ai truy cập**, lần vào lại đầu tiên sau đó mất khoảng 30–60 giây để "thức dậy" (cold start) — không phải app bị lỗi.

## Mô hình

- **Đặc trưng đầu vào:** HOG (8100 chiều) trích từ ảnh grayscale 128×128
- **Pipeline:** `StandardScaler` → `PCA` (giữ lại phần lớn phương sai) → `LogisticRegression`
- **Tinh chỉnh:** `GridSearchCV` trên số thành phần PCA và hệ số regularization `C`
- **Ngưỡng quyết định:** dự đoán chỉ được chấp nhận là Chó/Mèo khi xác suất ≥ 75%, nếu không sẽ trả về "Không xác định" thay vì đoán bừa
