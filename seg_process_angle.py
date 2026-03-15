from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO  
from skimage.morphology import skeletonize

# 載入預訓練的YOLOv8實例分割模型
model = YOLO('./best.pt')

def get_new_folder_name(base_path, base_name):
    i = 1
    while True:
        folder_name = base_name if i == 1 else f"{base_name}{i}"
        full_path = base_path / folder_name
        if not full_path.exists():
            return folder_name
        i += 1

def compute_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Lines are parallel

    intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    return int(intersect_x), int(intersect_y)

def compute_angle(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # 计算向量
    vector1 = np.array([x2 - x1, y2 - y1])
    vector2 = np.array([x4 - x3, y4 - y3])

    # 计算向量的长度
    length1 = np.linalg.norm(vector1)
    length2 = np.linalg.norm(vector2)

    if length1 == 0 or length2 == 0:
        return None  # 如果向量长度为0，无法计算角度

    # 计算向量的点积
    dot_product = np.dot(vector1, vector2)

    # 计算向量之间的角度（以弧度为单位）
    angle = np.arccos(dot_product / (length1 * length2))

    # 将角度转换为度
    angle_degrees = np.degrees(angle)
    print(f'箱尺角度：{angle_degrees:.1f}')

    return angle_degrees

def dynamic_morphology_filter_size(image):
    height, width = image.shape[:2]
    max_dim = max(height, width)

    if max_dim > 2400:
        return 300
    elif max_dim > 2000:
        return 250
    elif max_dim > 1600:
        return 200
    else:
        return 150

################################## 主程式開始 ##################################

def seg_process_angle(files, base_output_dir, script_dir, image_path):
    # 從路徑中提取文件名
    image_name = image_path.stem
    
    # 讀取原始影像
    orig_img = cv2.imread(str(image_path))

    # 對圖像進行推理(實例分割)
    results = model(str(image_path))

    # 創建輸出目錄
    base_output_dir.mkdir(parents=True, exist_ok=True)

    folder_name = get_new_folder_name(base_output_dir, 'predict')
    output_dir = base_output_dir / folder_name
    output_dir.mkdir()

    # 保存原始影像
    orig_img_path = str(output_dir / f"{image_name}_orig.png")
    cv2.imwrite(orig_img_path, orig_img)
    files["原始影像"] = f"{folder_name}/{image_name}_orig.png"

    # 創建標記的影像
    marked_image = orig_img.copy()
    combined_lines_image = np.zeros_like(marked_image)
    all_objects_image = np.zeros_like(marked_image)
    all_lines = []

    for i, result in enumerate(results):
        # 獲取所有預測的掩碼
        masks = result.masks.data.cpu().numpy()

        # 獲取所有預測的類別
        classes = result.boxes.cls.cpu().numpy()

        for j, (mask, cls) in enumerate(zip(masks, classes)):
            if result.names[int(cls)] == 'Grade-Rod':
                # 調整掩碼大小以匹配原始圖像
                resized_mask = cv2.resize(mask, (marked_image.shape[1], marked_image.shape[0]))

                # 在原始圖像上繪製掩碼輪廓
                contours, _ = cv2.findContours((resized_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(marked_image, contours, -1, (0, 255, 0), 2)

                # 構建一個半透明黑色遮罩
                mask_overlay = np.zeros_like(marked_image, dtype=np.uint8)
                non_object_mask = (resized_mask == 0)
                alpha = 0.5  # 透明度       
                for c in range(3):  # 對於每個顏色通道
                    mask_overlay[:, :, c] = marked_image[:, :, c] * alpha
                    marked_image[:, :, c] = np.where(non_object_mask, 
                                                        mask_overlay[:, :, c], 
                                                        marked_image[:, :, c])

                # 將物件添加到all_objects_image
                for c in range(3):
                    all_objects_image[:, :, c] = np.where(resized_mask > 0, orig_img[:, :, c], all_objects_image[:, :, c])

    # 保存標記的影像
    marked_image_path = str(output_dir / f"{image_name}_marked.png")
    cv2.imwrite(marked_image_path, marked_image)
    files["實例分割影像"] = f"{folder_name}/{image_name}_marked.png"

    # 保存所有物件的合成影像
    all_objects_image_path = str(output_dir / f"{image_name}_all_objects.png")
    cv2.imwrite(all_objects_image_path, all_objects_image)
    files["所有箱尺物件影像"] = f"{folder_name}/{image_name}_all_objects.png"

    white_area=0
    for i, result in enumerate(results):
        # 獲取所有預測的掩碼
        masks = result.masks.data.cpu().numpy()

        # 獲取所有預測的類別
        classes = result.boxes.cls.cpu().numpy()
        
        for j, (mask, cls) in enumerate(zip(masks, classes)):
            if result.names[int(cls)] == 'Grade-Rod':
                # 調整掩碼大小以匹配原始圖像
                resized_mask = cv2.resize(mask, (orig_img.shape[1], orig_img.shape[0]))

                # 創建全黑背景圖像（與原始圖像大小相同）
                object_img = np.zeros_like(orig_img)

                # 將掩碼應用到原始圖像上
                for c in range(3):
                    object_img[:, :, c] = np.where(resized_mask > 0, orig_img[:, :, c], object_img[:, :, c])

                # 保存分割結果
                object_img_path = str(output_dir / f"{image_name}_object_{i+1}_{j+1}_segmented.png")
                cv2.imwrite(object_img_path, object_img)
                files[f"箱尺物件{j+1}"] = f"{folder_name}/{image_name}_object_{i+1}_{j+1}_segmented.png"

                # 將分割結果進行二值化處理
                gray_object_img = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)
                _, binary_object_img = cv2.threshold(gray_object_img, 1, 255, cv2.THRESH_BINARY)

                # 動態調整形態學運算的 kernel 大小
                filter_size = dynamic_morphology_filter_size(orig_img)
                kernel = np.ones((5, 5), np.uint8)
                opening = cv2.morphologyEx(binary_object_img, cv2.MORPH_OPEN, kernel)
                kernel = np.ones((filter_size, filter_size), np.uint8)
                closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
                
                #計算白色部分面積
                white_area += np.sum(closing == 255)
                
                # 進行骨架化處理
                skeleton = skeletonize(closing // 255).astype(np.uint8) * 255

                # 保存形態學處理和骨架化後的結果
                morph_object_img_path = str(output_dir / f"{image_name}_object_{i+1}_{j+1}_morph.png")
                cv2.imwrite(morph_object_img_path, closing)
                files[f"物件{j+1}：二值化+形態學操作"] = f"{folder_name}/{image_name}_object_{i+1}_{j+1}_morph.png"

                skeleton_img_path = str(output_dir / f"{image_name}_object_{i+1}_{j+1}_skeleton.png")
                cv2.imwrite(skeleton_img_path, skeleton)
                files[f"物件{j+1}：骨架化"] = f"{folder_name}/{image_name}_object_{i+1}_{j+1}_skeleton.png"

                # 使用霍夫變換進行直線偵測
                lines = cv2.HoughLinesP(skeleton, 1, np.pi / 180, threshold=120, minLineLength=80, maxLineGap=20)

                if lines is not None:
                    # 計算所有直線的平均值，生成一條平滑的直線
                    avg_line = np.mean(lines, axis=0).astype(np.int32).flatten()
                    x1, y1, x2, y2 = avg_line

                    # 根據骨架圖像的尺寸延長直線
                    def extend_line(x1, y1, x2, y2, length):
                        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        if line_length == 0:
                            return x1, y1, x2, y2
                        extend_ratio = length / line_length
                        new_x1 = int(x1 - (x2 - x1) * extend_ratio)
                        new_y1 = int(y1 - (y2 - y1) * extend_ratio)
                        new_x2 = int(x2 + (x2 - x1) * extend_ratio)
                        new_y2 = int(y2 + (y2 - y1) * extend_ratio)
                        return new_x1, new_y1, new_x2, new_y2

                    extended_x1, extended_y1, extended_x2, extended_y2 = extend_line(x1, y1, x2, y2, max(skeleton.shape))

                    # 將平滑直線加入all_lines
                    all_lines.append([extended_x1, extended_y1, extended_x2, extended_y2])

                    # 繪製這條延長的平滑直線到combined_lines_image
                    cv2.line(combined_lines_image, (extended_x1, extended_y1), (extended_x2, extended_y2), (0, 255, 0), 2)

    # 計算並繪製所有直線的交點和角度
    intersection_points = []
    for idx1, line1 in enumerate(all_lines):
        for idx2, line2 in enumerate(all_lines):
            if idx1 >= idx2:
                continue
            intersection = compute_intersection(line1, line2)
            if intersection:
                intersection_points.append(intersection)
                cv2.circle(combined_lines_image, intersection, 5, (0, 0, 255), -1)
                
                # 計算並顯示角度
                angle = compute_angle(line1, line2)
                if angle:
                    angle_text = f"{angle:.2f}-degree"
                    cv2.putText(combined_lines_image, angle_text, intersection, cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10, cv2.LINE_AA)

    # 保存所有直線的合成影像
    combined_lines_image_path = str(output_dir / f"{image_name}_combined_lines.png")
    cv2.imwrite(combined_lines_image_path, combined_lines_image)
    files["合成兩直線影像 與 計算交接角度"] = f"{folder_name}/{image_name}_combined_lines.png"

    return files, all_objects_image, intersection_points, white_area