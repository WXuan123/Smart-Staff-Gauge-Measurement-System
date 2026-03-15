import cv2
import numpy as np
from ultralytics import YOLO

# 載入數字辨識模型
num_model = YOLO("number.pt")  # pretrained YOLOv8n model

def check_red_color(img, box_2, box_1):
    # 將BGR圖像轉換到HSV色彩空間
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 設定紅色的範圍，包括暗紅和亮紅
    # 暗紅色調
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    # 亮紅色調
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    # 創建紅色的遮罩
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # 應用遮罩
    red_detection = cv2.bitwise_and(img, img, mask=mask_red)
    # 確保切片索引為整數
    x1, y1 = int(box_2[0]), int(box_2[1])
    x2, y2 = int(box_1[2]), int(box_1[3])
    # 檢查指定區域內是否有紅色
    red_area = red_detection[y1:y2, x1:x2]
    if np.any(red_area):
        print("有紅點")
        return 1
    else:
        print("沒有紅點")
        return 0

def detect_circles_above_number(img, box_2, box_1):
    # 讀取圖像並轉為灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 2)
    
    # 進行圓形偵測
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 15,
                               param1=50, param2=30, minRadius=0, maxRadius=50)
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # 計算位於指定框上方的圓形數量
        for circle in circles[0, :]:
            x_center, y_center, radius = circle
            # 檢查圓心是否在數字的上方
            if box_2[0] <= x_center <= box_2[2] and y_center < box_2[1] and y_center > box_1[3]:
                count += 1
                print(f'[{x_center},{y_center}]')

        if count > 0:
            return count
        else:
            count+=check_red_color(img, box_2, box_1)
            return count
    else:
        count+=check_red_color(img, box_2, box_1)
        return count
    
def number(image_path):
    names=[8,5,4,9,1,7,6,3,2,10]  # class names

    results = num_model(image_path)  # return a list of Results objects
    for r in results:
        new_image_path = image_path.replace(".png", "_number.png")  # 新的圖片檔名
        r.save(filename=new_image_path)  # Save annotated image as a new file

        if r.boxes.xyxy.numel() > 0: # Check if there are any detection boxes
            # 找到image中心點(箱尺交點)
            img = cv2.imread(image_path)
            img_height, img_width = img.shape[:2]
            img_center_y = img_height / 2

            # Create a list to hold distances, boxes and classes
            every_boxes = []

            for i in range(r.boxes.xyxy.size(0)):
                # Convert tensor to numpy array, select the coordinates of the current detection box
                box = r.boxes.xyxy[i].cpu().numpy()
                # Get the class of the current detection box
                cls = int(r.boxes.cls[i].item())
                class_name=names[cls]
                # Calculate the center of the bounding box
                box_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                # Calculate distance from the image center
                distance = np.linalg.norm(box_center - [img_width / 2, img_center_y])
                # Append distance, box, and class to the list
                every_boxes.append((distance, box, class_name))

            # Sort the list by distance
            every_boxes.sort(key=lambda x: x[0])

            # 找到兩個最近的數字
            closest_boxes = every_boxes[:2]
            # 確保列表中至少有兩個元素
            if len(closest_boxes) >= 2:
                if closest_boxes[0][2] == closest_boxes[1][2]:  # Check if the classes of the two closest boxes are the same
                    if len(every_boxes) > 2:  # 確保還有第三個 box 可以取代
                        closest_boxes[1] = every_boxes[2]  # Replace the second box with the third closest box
                    
            # 將數字由大到小排列
            closest_boxes.sort(key=lambda x: x[2], reverse=True)
            if closest_boxes[0][2] == 10 and closest_boxes[1][2] <= 2:
                closest_boxes.sort(key=lambda x: x[2])

            # Print兩數字頂端的座標
            for _, box, cls in closest_boxes:
                print(f"Coordinates: [{box[0]:.4f}, {box[1]:.4f}, {box[2]:.4f}, {box[3]:.4f}], Class: {cls}")
            
            # Calculate數字頂端到交點的y距離
            top_differences = []
            for _, box, cls in closest_boxes:
                box_top_y = box[1]
                difference_top = box_top_y - img_center_y #可能是負的
                top_differences.append(difference_top)

            # Calculatex兩數字頂端到交點的y距離的比例
            if len(top_differences) == 2:
                if((top_differences[0] * top_differences[1]) <= 0): #判斷是否"一個數字在上一個數字在下"
                    top_differences[0]=abs(top_differences[0])
                    top_differences[1]=abs(top_differences[1])
                    ratio = top_differences[1] / (top_differences[0] + top_differences[1])
                else:
                    print("交點在下面!")
                    ratio = (-1) * top_differences[1] / (top_differences[0] - top_differences[1])
                print(f"Ratio of the top height differences: {ratio:.2f}")
                depth = (closest_boxes[1][2] * 10) + (ratio * 10)
                # 圓點檢測(百位數)
                count_circles = detect_circles_above_number(img,closest_boxes[1][1],closest_boxes[0][1])
                print(f'圓形數量: {count_circles}')
            elif len(top_differences) == 1: # 如果畫面中只有一個數字
                depth = ((closest_boxes[0][2] - 1) * 10) + 5
                count_circles = detect_circles_above_number(img,[closest_boxes[0][1][0],0,closest_boxes[0][1][2],0],closest_boxes[0][1])

            depth+=count_circles*100
            print(f'刻度: {depth:.1f} cm')

            # 在圖片右上角顯示depth值
            text = f'{depth:.1f} cm'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            font_color = (0, 0, 255)  # 紅色
            thickness = 4
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = img_width - text_size[0] - 10
            text_y = text_size[1] + 10
            cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, thickness)

            # 儲存新圖片
            final_image_path = image_path.replace(".png", "_depth.png")
            cv2.imwrite(final_image_path, img)