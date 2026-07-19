from flask import Flask, request, render_template
import joblib
import cv2
import numpy as np
import base64
from skimage.feature import hog

app = Flask(__name__)

# Model được lưu bằng joblib.dump(best_model, ...) trong notebook,
# nên PHẢI load bằng joblib.load (không dùng pickle.load trực tiếp).
model = joblib.load('pipeline_catdog_model_final.pkl')

# Các thông số này phải khớp CHÍNH XÁC với lúc train (xem cell 9 / 21 trong notebook)
IMG_SIZE = 128

# Ngưỡng tin cậy tối thiểu để chấp nhận kết quả (giống notebook, cell predict_and_show).
# Nếu xác suất của cả 2 lớp đều dưới ngưỡng này -> trả về "Unknown".
UNKNOWN_THRESHOLD = 0.75


def resize_with_padding(img, target_size):
    """Resize giữ nguyên tỉ lệ khung hình, pad thêm viền đen cho đủ kích thước.

    PHẢI giống hệt hàm cùng tên trong notebook train, nếu không dự đoán sẽ bị lệch.
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size, target_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h))

    if img.ndim == 3:
        canvas = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
    else:
        canvas = np.zeros((target_h, target_w), dtype=img.dtype)

    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas


def preprocess_image_bytes(img_bytes):
    """Tiền xử lý ảnh từ bytes (upload qua Flask), đúng như lúc train:
    resize+pad -> grayscale -> trích xuất đặc trưng HOG.
    """
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Không đọc được ảnh từ dữ liệu upload.")

    img_pad = resize_with_padding(img, IMG_SIZE)
    img_gray = cv2.cvtColor(img_pad, cv2.COLOR_BGR2GRAY)

    features = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    features = features.astype(np.float32).reshape(1, -1)
    return features


@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    result_type = None  # 'dog' | 'cat' | 'unknown' -> dùng để tô màu thẻ kết quả
    img_data = None
    img_mime = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="Không tìm thấy file!")

        file = request.files['file']
        if file.filename != '':
            img_bytes = file.read()
            img_data = base64.b64encode(img_bytes).decode('utf-8')
            img_mime = file.content_type or 'image/jpeg'

            # Tiền xử lý & dự đoán (model tự áp dụng StandardScaler + PCA trước khi Logistic phân loại)
            features = preprocess_image_bytes(img_bytes)
            probs = model.predict_proba(features)[0]
            cat_prob, dog_prob = probs[0], probs[1]

            # Áp dụng ngưỡng tin cậy, giống hệt logic trong notebook (predict_and_show)
            if dog_prob >= UNKNOWN_THRESHOLD:
                result = "Đây là Chó"
                result_type = "dog"
            elif cat_prob >= UNKNOWN_THRESHOLD:
                result = "Đây là Mèo"
                result_type = "cat"
            else:
                result = "Không nhận dạng được đây là Chó hay Mèo"
                result_type = "unknown"

    return render_template(
        'index.html',
        result=result,
        result_type=result_type,
        img_data=img_data,
        img_mime=img_mime,
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)