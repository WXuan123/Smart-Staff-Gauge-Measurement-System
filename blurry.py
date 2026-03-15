import cv2
from PIL import Image

def is_blurry(image_path, threshold=500.0):
    """
    判斷影像是否模糊
    :param image_path: 影像路徑
    :param threshold: 判斷模糊的閾值，數值越小表示容忍的模糊程度越低
    :return: 布林值，True 表示影像清晰，False 表示影像模糊
    """
    image_path = str(image_path)  # Ensure it's a string
    # 讀取影像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"影像檔案 '{image_path}' 不存在。")

    # 轉換為灰度影像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 計算 Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance_of_laplacian = laplacian.var()

    print(f"Laplacian 方差: {variance_of_laplacian:.2f}")

    # 判斷是否模糊
    return variance_of_laplacian > threshold

def resolution(image_path):
    """
    判斷影像解析度是否至少為500x500像素
    :image_path: 影像檔案的路徑。
    :return:如果影像的長和寬都至少為960像素，返回True；否則返回False。
    """
    image_path = str(image_path)  # Ensure it's a string
    try:
        # 打開影像檔案
        with Image.open(image_path) as img:
            # 獲取影像的解析度
            width, height = img.size
            print(f"解析度：{width} * {height}")
            # 判斷解析度是否符合要求
            if width*height < 921600:
                return False
            return True
        
    except IOError:
        print("無法打開影像檔案。請確保檔案路徑正確並且檔案格式支持。")
        return False
    
def blurry(image_path):
    clear = is_blurry(image_path)
    size = resolution(image_path)

    if clear and size:
        print("影像是清晰的")
    elif size:
        print("影像是模糊的")
    else:
        print("影像解析度太小")
