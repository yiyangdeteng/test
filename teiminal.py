import cv2
import numpy as np
import os
from datetime import datetime

# ==================== 只需修改这里的输入路径 ====================
INPUT_IMAGE = "4.jpg"  # 修改为您的图片路径
# ==============================================================

def extract_painting(input_path):
    """提取画作的主函数"""
    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        print(f"错误: 无法加载图像 {input_path}")
        return False
    
    # 预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 50, 200)
    
    # 查找轮廓
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # 寻找四边形轮廓
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            # 透视变换
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            # 计算变换后的尺寸
            (tl, tr, br, bl) = rect
            width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
            height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
            
            # 执行透视变换
            dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            result = cv2.warpPerspective(img, M, (width, height))
            
            # 保存结果
            os.makedirs("output", exist_ok=True)
            output_path = f"output/extracted_{os.path.basename(input_path)}"
            cv2.imwrite(output_path, result)
            print(f"成功: 结果已保存到 {output_path}")
            return True
    
    print("未检测到合适的画框轮廓")
    return False

# 运行程序
if __name__ == "__main__":
    extract_painting(INPUT_IMAGE)