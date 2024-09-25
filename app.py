from flask import Flask, request, render_template
import cv2
import numpy as np
from retinaface import RetinaFace
import base64

app = Flask(__name__)

# 重ねる画像を読み込みます（透過PNG推奨）
overlay_img = cv2.imread('overlay.png', cv2.IMREAD_UNCHANGED)

def overlay_on_faces(img, overlay_img, offset_y_ratio=0.2):
    # 顔検出
    faces = RetinaFace.detect_faces(img)

    # 画像の高さと幅を取得
    img_h, img_w = img.shape[:2]

    # 顔に画像を重ねる
    if isinstance(faces, dict):
        for face in faces.values():
            x1, y1, x2, y2 = face['facial_area']
            face_w, face_h = x2 - x1, y2 - y1

            # 顔の中心座標を計算
            cx = x1 + face_w // 2
            cy = y1 + face_h // 2

            # 重ねる画像の新しいサイズを計算（1.5倍）
            new_w = int(face_w * 1.5)
            new_h = int(face_h * 1.5)

            # 新しい左上の座標を計算
            x1_new = cx - new_w // 2
            y1_new = cy - new_h // 2

            # 位置を上にずらす（オフセットを適用）
            offset_y = int(new_h * offset_y_ratio)
            y1_new -= offset_y
            y2_new = y1_new + new_h
            x2_new = x1_new + new_w

            # 画像が範囲外に出ないように調整
            x1_new = max(0, x1_new)
            y1_new = max(0, y1_new)
            x2_new = min(img_w, x2_new)
            y2_new = min(img_h, y2_new)

            # 調整後の幅と高さ
            adjusted_w = x2_new - x1_new
            adjusted_h = y2_new - y1_new

            # 重ねる画像をリサイズ
            overlay_resized = cv2.resize(overlay_img, (adjusted_w, adjusted_h))

            # アルファブレンド処理
            if overlay_resized.shape[2] == 4:  # 透過PNGの場合
                alpha_s = overlay_resized[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                for c in range(0, 3):
                    img[y1_new:y2_new, x1_new:x2_new, c] = (
                        alpha_s * overlay_resized[:, :, c] +
                        alpha_l * img[y1_new:y2_new, x1_new:x2_new, c]
                    )
            else:
                img[y1_new:y2_new, x1_new:x2_new] = overlay_resized

    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # アップロードされた画像を読み込み
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # 顔に画像を重ねる処理を実行（オフセット調整可能）
        result_img = overlay_on_faces(img, overlay_img, offset_y_ratio=0.1)

        # 画像をエンコードしてHTMLに埋め込み
        _, buffer = cv2.imencode('.jpg', result_img)
        img_str = base64.b64encode(buffer).decode('utf-8')

        return render_template('index.html', result=img_str)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
