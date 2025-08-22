import cv2
import numpy as np
import argparse
import os

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """
    调整图像亮度和对比度
    """
    image = np.float32(image)
    image = image * (1 + contrast/127.0) - contrast + brightness
    image = np.clip(image, 0, 255)
    return np.uint8(image)

def preprocess_image(image):

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 增强亮度和对比度以便更好地检测边缘
    gray = adjust_brightness_contrast(gray, brightness=30, contrast=30)
    # 使用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def find_edges(image):
    """
    查找图像中的边缘
    """
    # 使用自适应阈值处理，适用于光照不均匀的情况
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    # 边缘检测
    edges = cv2.Canny(thresh, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    return edges

def find_largest_quadrilateral(edges, original_image):
    """
    在边缘图像中寻找最大的四边形
    """
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 按面积排序轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # 寻找近似四边形的轮廓
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        # 如果是四边形，返回它
        if len(approx) == 4:
            # 验证四边形是否合理（面积不能太小）
            area = cv2.contourArea(approx)
            if area > original_image.shape[0] * original_image.shape[1] * 0.1:  # 至少占原图面积的10%
                return approx
    return None

def order_points(pts):
    """
    对四个点进行排序：左上，右上，右下，左下
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def auto_crop_border(image, border_color_threshold=30):
    """
    自动裁剪图像边缘的有色边框（非黑色）
    """
    # 转换为灰度图或保留彩色
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    # 将接近边框颜色的区域设为白色，其他为黑色
    _, thresh = cv2.threshold(gray, border_color_threshold, 255, cv2.THRESH_BINARY_INV)
    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 获取最大轮廓的边界矩形
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt) 
        # 裁剪图像
        cropped = image[y:y+h, x:x+w]
        return cropped 
    return image

def perspective_correction(image_path, output_path):
    """
    主函数：执行透视矫正和裁剪
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return False
    orig = image.copy()
    print("正在处理图像...") 
    # 预处理图像
    processed = preprocess_image(image)
    # 查找边缘
    edges = find_edges(processed)
    # 查找最大的四边形
    quadrilateral = find_largest_quadrilateral(edges, image) 
    if quadrilateral is None:
        print("未找到合适的四边形，尝试替代方法...")
        # 尝试使用霍夫直线检测来寻找边界
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10) 
        if lines is not None:
            # 创建一个空白图像绘制所有直线
            line_image = np.zeros_like(image)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            # 从直线图像中再次查找轮廓
            line_gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
            quadrilateral = find_largest_quadrilateral(line_gray, image) 
    if quadrilateral is None:
        print("无法自动检测到边界，将返回原始图像")
        cv2.imwrite(output_path, image)
        return True
    # 确保四边形是凸的
    if not cv2.isContourConvex(quadrilateral):
        print("找到的四边形不是凸的，尝试优化...")
        # 计算凸包
        hull = cv2.convexHull(quadrilateral)
        if len(hull) == 4:
            quadrilateral = hull
        else:
            print("无法优化为凸四边形，将使用原始检测结果") 
    # 对点进行排序
    rect = order_points(quadrilateral.reshape(4, 2)) 
    # 计算新图像的宽度和高度
    (tl, tr, br, bl) = rect 
    # 计算宽度
    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    max_width = max(int(width_top), int(width_bottom)) 
    # 计算高度
    height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    max_height = max(int(height_left), int(height_right))
    # 定义目标点
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32") 
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst) 
    # 应用透视变换
    warped = cv2.warpPerspective(orig, M, (max_width, max_height))
    # 自动裁剪黑色边框
    final = auto_crop_border(warped) 
    # 保存结果
    cv2.imwrite(output_path, final)
    print(f"处理完成，结果已保存到: {output_path}") 
    return True

def process_directory(input_dir, output_dir):
    """
    处理目录中的所有图像
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"corrected_{filename}")
            print(f"处理: {filename}")
            success = perspective_correction(input_path, output_path)
     
            if success:
                print(f"成功处理: {filename}")
            else:
                print(f"处理失败: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='自动透视矫正和裁剪美术馆照片')
    parser.add_argument('--input', '-i', help='输入图像路径或目录')
    parser.add_argument('--output', '-o', help='输出图像路径或目录') 
    args = parser.parse_args()
    if args.input and args.output:
        if os.path.isdir(args.input):
            # 处理整个目录
            process_directory(args.input, args.output)
        else:
            # 处理单个文件
            success = perspective_correction(args.input, args.output)
            if success:
                print("处理完成!")
            else:
                print("处理失败!")
    else:
        input_image = "02 (2).jpg"  # 默认输入图像名称
        output_image = "corrected_photo.jpg"  # 默认输出图像名称
        
        if os.path.exists(input_image):
            success = perspective_correction(input_image, output_image)
            if success:
                print("处理完成!")
            else:
                print("处理失败!")
        else:
            print(f"请提供输入图像路径，或将图像命名为 '{input_image}' 放在当前目录")