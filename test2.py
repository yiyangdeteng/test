import os
import cv2
import numpy as np

# 配置参数
PROCESSED_PATH = "./test"     # 矫正后图片输出路径
CORRECTED_SIZE = (400, 400)      # 矫正后固定尺寸

def init_folders():
    """初始化输出文件夹"""
    if not os.path.exists(PROCESSED_PATH):
        os.makedirs(PROCESSED_PATH)
        print(f"创建文件夹: {PROCESSED_PATH}")

def order_points(pts):
    """角点排序"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    return rect

def four_point_transform(image, pts):
    """四点透视变换"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # 计算宽高（保留原始比例）
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    
    # 目标点定义
    dst = np.array([
        [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    
    # 透视变换
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (int(width), int(height)))

def adaptive_preprocess(image, method):
    """自适应预处理（多场景适配）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == 0:  # 标准处理
        gray = cv2.equalizeHist(gray)
        return cv2.GaussianBlur(gray, (5, 5), 0)
    elif method == 1:  # 高对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        return cv2.bilateralFilter(gray, 9, 75, 75)
    elif method == 2:  # 低光照增强
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        gray = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        return cv2.medianBlur(gray, 5)
    elif method == 3:  # 高噪声处理
        gray = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)
        return cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def detect_quadrilateral(img, debug=False):
    """改进的四边形检测"""
    best_contour = None
    best_score = -1
    selected_method = -1
    
    # 多预处理方法+多尺度边缘检测
    for method in range(4):
        processed = adaptive_preprocess(img, method)
        for sigma in [0.33, 0.66, 1.0]:  # 多尺度Canny参数
            v = np.median(processed)
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edged = cv2.Canny(processed, lower, upper)
            
            # 形态学强化边缘
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # 筛选凸四边形轮廓
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < img.shape[0]*img.shape[1]*0.05:  # 过滤小区域
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
                
                # 凸四边形判定
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    x,y,w,h = cv2.boundingRect(approx)
                    aspect_ratio = w / float(h)
                    area_ratio = area / float(w*h)
                    # 评分机制：面积+矩形相似度+长宽比合理性
                    score = area + 1000*min(area_ratio, 1.0) + 500*(1 - abs(1 - aspect_ratio))
                    if score > best_score:
                        best_score = score
                        best_contour = approx
                        selected_method = method  # 记录最佳预处理方法
    
    if debug and selected_method != -1:
        print(f"最佳预处理方法: {selected_method}")
    
    return best_contour

def remove_frame_artifacts(image):
    """去除边框残留"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    contours, _ = cv2.findContours(255-mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x,y,w,h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return image[y:y+h, x:x+w]
    return image

def perspective_correction(img_path, output_path=None, debug=False):
    """透视矫正主函数"""
    image = cv2.imread(img_path)
    if image is None:
        print(f"错误：无法读取图像 {img_path}")
        return None, None
    
    orig = image.copy()
    h, w = image.shape[:2]
    # 图像缩放（避免大尺寸卡顿）
    scale = min(1600/w, 1600/h) if max(w,h) > 1600 else 1.0
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale)
    
    # 自动检测四边形
    screenCnt = detect_quadrilateral(image, debug)
    
    # 自动检测失败则返回None
    if screenCnt is None:
        print("自动检测失败，无法完成矫正")
        return None, None
    
    # 执行透视变换
    try:
        pts = screenCnt.reshape(4,2).astype("float32")
        if scale != 1.0:  # 还原缩放后的坐标
            pts = pts / scale
        # 透视变换+去边框残留
        warped = four_point_transform(orig, pts)
        warped = remove_frame_artifacts(warped)
        # 调整为固定尺寸
        warped = cv2.resize(warped, CORRECTED_SIZE)
        
        # 调试可视化
        if debug:
            debug_img = orig.copy()
            cv2.drawContours(debug_img, [screenCnt.astype(int)], -1, (0,255,0), 3)
            cv2.imshow("检测结果", cv2.resize(debug_img, (800,600)))
            cv2.imshow("矫正结果", cv2.resize(warped, (800,600)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # 保存矫正结果
        if output_path:
            cv2.imwrite(output_path, warped)
            print(f"矫正结果保存至: {output_path}")
        return warped, output_path
    
    except Exception as e:
        print(f"矫正失败: {str(e)}")
        return None, None

# 测试代码
if __name__ == "__main__":
    init_folders()
    
    # 测试图片路径（请替换为实际图片路径）
    test_image_path = "02 (2).jpg"
    
    # 输出路径
    output_filename = os.path.basename(test_image_path).split('.')[0] + "_corrected.jpg"
    output_path = os.path.join(PROCESSED_PATH, output_filename)
    
    # 执行透视矫正（debug=True将显示中间结果）
    corrected_img, saved_path = perspective_correction(
        test_image_path, 
        output_path=output_path, 
        debug=True
    )

    if corrected_img is not None:
        print("透视矫正成功！")
    else:
        print("透视矫正失败！")