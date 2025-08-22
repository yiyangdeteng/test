import cv2
import numpy as np
import os

# 全局变量用于手动选择点
manual_points = []
selected_method = -1  # 记录最佳预处理方法

def click_event(event, x, y, flags, params):
    """鼠标点击事件处理函数，用于手动选择角点"""
    global manual_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(manual_points) < 4:
            manual_points.append((x, y))
            cv2.circle(params[0], (x, y), 5, (0, 0, 255), -1)
            cv2.putText(params[0], str(len(manual_points)), 
                       (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.imshow(params[1], params[0])

def order_points(pts):
    """对四边形的四个点进行排序（左上、右上、右下、左下）"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    return rect

def four_point_transform(image, pts):
    """应用四点透视变换"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # 计算新图像的宽度和高度
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    
    # 构建目标点
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")
    
    # 计算透视变换矩阵并应用
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (int(width), int(height)))
    
    return warped

def adaptive_preprocess(image, method):
    """自适应预处理方法"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == 0:  # 标准处理
        gray = cv2.equalizeHist(gray)
        return cv2.GaussianBlur(gray, (5, 5), 0)
    
    elif method == 1:  # 高对比度处理
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

def detect_quadrilateral(image, debug=False):
    """改进的四边形检测算法"""
    global selected_method
    best_contour = None
    best_score = -1
    
    # 尝试不同的预处理方法
    for method in range(4):
        processed = adaptive_preprocess(image, method)
        
        # 多尺度边缘检测
        for sigma in [0.33, 0.66, 1.0]:
            v = np.median(processed)
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edged = cv2.Canny(processed, lower, upper)
            
            # 形态学操作强化边缘
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # 寻找轮廓
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 筛选轮廓
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < image.shape[0]*image.shape[1]*0.05:  # 忽略太小的区域
                    continue
                
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
                
                # 检查是否为凸四边形
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    # 计算轮廓评分
                    x,y,w,h = cv2.boundingRect(approx)
                    aspect_ratio = w / float(h) if h != 0 else 0
                    area_ratio = area / float(w*h) if (w*h) != 0 else 0
                    
                    # 评分标准
                    score = area + 1000*min(area_ratio, 1.0) + 500*(1 - abs(1 - aspect_ratio))
                    
                    if score > best_score:
                        best_score = score
                        best_contour = approx
                        selected_method = method  # 记录最佳预处理方法
        
        if debug:
            cv2.destroyWindow(f"Method {method}")  # 不显示预处理中间窗口
    
    return best_contour

def remove_frame_artifacts(image):
    """去除矫正后图像的残留边框"""
    if image is None:
        return None
    # 转换为HSV空间检测黑色边框
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # 找到非黑色区域
    contours, _ = cv2.findContours(255-mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x,y,w,h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return image[y:y+h, x:x+w]
    return image

def perspective_correction(image_path, output_path=None, debug=False, manual_mode=False):
    """改进的透视矫正主函数"""
    global manual_points, selected_method
    manual_points = []  # 重置手动点
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        return None  # 读取失败返回None
    
    orig = image.copy()
    h, w = image.shape[:2]
    
    # 调整图像大小（保持比例）
    scale = min(1600/w, 1600/h) if max(w,h) > 1600 else 1.0
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale)
    
    screenCnt = None
    if not manual_mode:
        screenCnt = detect_quadrilateral(image, debug)
    
    # 手动模式处理
    if screenCnt is None or manual_mode:
        window_name = "选择四个角点 (左上→右上→右下→左下)"
        temp_img = image.copy()
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, click_event, [temp_img, window_name])
        
        while True:
            cv2.imshow(window_name, temp_img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键退出
                cv2.destroyWindow(window_name)
                return None
            if len(manual_points) == 4:
                break
        
        cv2.destroyWindow(window_name)  # 关闭手动选择窗口
        
        if len(manual_points) == 4:
            screenCnt = np.array(manual_points, dtype="float32").reshape((4,1,2))
        else:
            return None  # 未选择足够的点
    
    # 执行矫正
    if screenCnt is not None:
        try:
            pts = screenCnt.reshape(4,2).astype("float32")
            if scale != 1.0 and not manual_mode:
                pts = pts / scale
                
            warped = four_point_transform(orig, pts)
            warped = remove_frame_artifacts(warped)  # 去除残留边框
            
            return warped  # 返回矫正结果
            
        except Exception as e:
            return None  # 矫正失败返回None
    return None  # 无有效轮廓返回None

if __name__ == "__main__":
    input_img = "34 (2).jpg"
    output_img = "output.jpg"
    
    # 先尝试自动检测
    result = perspective_correction(input_img, output_img)
    
    # 自动失败后尝试手动
    if result is None:
        result = perspective_correction(input_img, output_img, manual_mode=True)
    
    # 最终只显示result（矫正结果）
    if result is not None:
        # 调整显示大小（保持比例）
        h, w = result.shape[:2]
        max_dim = 800
        scale = min(max_dim/w, max_dim/h)
        if scale < 1:
            result = cv2.resize(result, None, fx=scale, fy=scale)
        
        cv2.imshow("矫正结果", result)
        cv2.waitKey(0)  # 等待用户按键关闭窗口
        cv2.destroyAllWindows()