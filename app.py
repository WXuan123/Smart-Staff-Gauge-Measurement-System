from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO  
from skimage.morphology import skeletonize
from PIL import Image
from blurry import blurry
from seg_process_angle import seg_process_angle
from number import number

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

def get_new_folder_name(base_path, base_name):
    i = 1
    while True:
        folder_name = base_name if i == 1 else f"{base_name}{i}"
        full_path = base_path / folder_name
        if not full_path.exists():
            return folder_name
        i += 1           

def crop_image_centered(image, center, size):
    x, y = center
    half_size = size // 2
    x1, y1 = max(0, x - half_size), max(0, y - half_size)
    x2, y2 = min(image.shape[1], x + half_size), min(image.shape[0], y + half_size)
    cropped_image = image[y1:y2, x1:x2]

    # 如果切出的影像小於指定大小，則用黑色填充
    if cropped_image.shape[0] != size or cropped_image.shape[1] != size:
        padded_image = np.zeros((size, size, 3), dtype=image.dtype)
        padded_image[:cropped_image.shape[0], :cropped_image.shape[1]] = cropped_image
        return padded_image
    return cropped_image

def add_transparency(image):
    # 將影像轉換為BGRA格式
    b, g, r = cv2.split(image)
    alpha = np.where((b == 0) & (g == 0) & (r == 0), 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r, alpha])

################################################## 網頁 ##################################################

# 獲取當前腳本的絕對路徑
script_dir = Path(__file__).resolve().parent

# 創建輸出目錄
base_output_dir = script_dir / 'runs' / 'segment'
base_output_dir.mkdir(parents=True, exist_ok=True)

folder_name = get_new_folder_name(base_output_dir, 'predict')
output_dir = base_output_dir / folder_name
output_dir.mkdir()

@app.route('/', methods=['GET', 'POST'])
def index():
    files = {}
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image_path = Path(app.config['UPLOAD_FOLDER']) / file.filename
            file.save(image_path)

            # 從路徑中提取文件名
            image_filename = image_path.stem

            # 讀取原始影像
            orig_img = cv2.imread(str(image_path))

            is_blurry=blurry(image_path) #從blurry.py引入

            files, all_objects_image, intersection_points, white_area = seg_process_angle(files, base_output_dir, script_dir, image_path)
                #從seg_process_angle.py引入

            # 計算箱尺所佔的影像面積比例
            height, width = all_objects_image.shape[:2]
            min_side = min(height, width)
            total_area = height * width
            black_area = total_area - white_area
            if black_area == 0:
                area_ratio = 1  # 避免除以零的錯誤
            else:
                area_ratio = white_area / black_area
            print(f'ratio={area_ratio}!!')

            #根據箱尺面積調整交點影像的大小
            if area_ratio > 0.2:
                focus_size = min_side
            elif area_ratio > 0.12 and area_ratio <= 0.2:
                focus_size = int(min_side // 2)
            elif area_ratio > 0.1 and area_ratio <= 0.12:
                focus_size = int(min_side // 2.5)
            elif area_ratio > 0.05 and area_ratio <= 0.1:
                focus_size = min_side // 3
            else:
                focus_size = min_side // 4

            # 以交點為中心，從所有物件影像中切出指定大小的影像
            for idx, point in enumerate(intersection_points):
                cropped_img = crop_image_centered(all_objects_image, point, focus_size)
                cropped_img_with_alpha = add_transparency(cropped_img)
                cropped_img_path = str(output_dir / f"{image_filename}_intersection_{idx+1}.png")
                cv2.imwrite(cropped_img_path, cropped_img_with_alpha)
                number(cropped_img_path) #從number.py引入
                files[f"焦點影像 {idx+1}"] = f"{folder_name}/{image_filename}_intersection_{idx+1}.png"
                files[f"數字辨識 {idx+1}"] = f"{folder_name}/{image_filename}_intersection_{idx+1}_number.png"
                files[f"刻度判斷 {idx+1}"] = f"{folder_name}/{image_filename}_intersection_{idx+1}_depth.png"

    return render_template('index.html', files=files)

@app.route('/runs/segment/<path:filename>')
def download_file(filename):
    return send_from_directory('runs/segment', filename)

if __name__ == '__main__':
    app.run(debug=True)