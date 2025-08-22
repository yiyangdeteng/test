import cv2
import numpy as np
import os
import logging
from datetime import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("frame_removal.log"),
        logging.StreamHandler()
    ]
)

class FrameRemover:
    def __init__(self):
        """初始化画框移除器"""
        self.detected_contours = []
        self.debug_mode = False
        self.manual_points = []
        
    def _preprocess_image(self, img, strategy="auto"):
        """多策略图像预处理"""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 根据图像特征自动选择预处理策略
        if strategy == "auto":
            # 分析图像亮度
            brightness = np.mean(gray)
            # 分析图像对比度
            contrast = gray.std()
            
            logging.debug(f"图像亮度: {brightness:.2f}, 对比度: {contrast:.2f}")
            
            if brightness < 50:  # 过暗图像
                strategy = "low_light"
            elif brightness > 200:  # 过亮图像
                strategy = "high_light"
            elif contrast < 30:  # 低对比度
                strategy = "low_contrast"
            else:  # 正常图像
                strategy = "normal"
        
        # 应用相应的预处理策略
        if strategy == "low_light":
            # 低光环境处理
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            processed = clahe.apply(gray)
            processed = cv2.GaussianBlur(processed, (3, 3), 0)
            processed = cv2.equalizeHist(processed)
            
        elif strategy == "high_light":
            # 高光环境处理
            processed = cv2.GaussianBlur(gray, (5, 5), 0)
            processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
        elif strategy == "low_contrast":
            # 低对比度处理
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            processed = cv2.GaussianBlur(processed, (3, 3), 0)
            
        else:  # normal
            # 正常图像处理
            processed = cv2.GaussianBlur(gray, (5, 5), 0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            processed = clahe.apply(processed)
        
        if self.debug_mode:
            cv2.imshow(f"预处理结果 ({strategy})", processed)
            cv2.waitKey(0)
            
        return processed
    
    def _detect_edges(self, processed_img):
        """多阈值边缘检测"""
        edges = []
        # 尝试多个阈值组合
        threshold_combinations = [
            (30, 100), (50, 150), (20, 80), (10, 50)
        ]
        
        for t1, t2 in threshold_combinations:
            edge = cv2.Canny(processed_img, t1, t2)
            edges.append(edge)
        
        # 合并边缘检测结果，取最强边缘
        combined_edges = np.max(edges, axis=0)
        
        # 形态学处理增强边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_DILATE, kernel, iterations=1)
        
        if self.debug_mode:
            cv2.imshow("合并边缘检测结果", combined_edges)
            cv2.waitKey(0)
            
        return combined_edges
    
    def _find_frame_contour(self, img, edges):
        """多策略寻找画框轮廓"""
        h, w = img.shape[:2]
        img_area = h * w
        frame_contour = None
        
        # 尝试不同的轮廓检测模式
        contour_modes = [cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_CCOMP]
        approx_epsilons = [0.015, 0.02, 0.025]  # 不同的多边形逼近精度
        
        for mode in contour_modes:
            contours, _ = cv2.findContours(edges.copy(), mode, cv2.CHAIN_APPROX_SIMPLE)
            
            # 筛选有效轮廓
            valid_contours = []
            for c in contours:
                area = cv2.contourArea(c)
                # 面积在图像面积的5%-95%之间
                if 0.05 * img_area < area < 0.95 * img_area:
                    # 计算轮廓周长
                    perimeter = cv2.arcLength(c, True)
                    if perimeter > 100:  # 过滤太小的轮廓
                        valid_contours.append((area, perimeter, c))
            
            # 按面积排序
            valid_contours.sort(reverse=True, key=lambda x: x[0])
            
            # 尝试不同的逼近精度寻找四边形
            for area, perimeter, contour in valid_contours[:10]:  # 检查前10个最大轮廓
                for eps in approx_epsilons:
                    approx = cv2.approxPolyDP(contour, eps * perimeter, True)
                    if len(approx) == 4:
                        # 验证是否为凸多边形
                        if cv2.isContourConvex(approx):
                            # 验证长宽比
                            x, y, w_rect, h_rect = cv2.boundingRect(approx)
                            aspect_ratio = w_rect / float(h_rect) if h_rect > 0 else 0
                            if 0.3 < aspect_ratio < 3.0:  # 合理的画框比例
                                frame_contour = approx
                                logging.debug(f"找到画框轮廓 - 面积: {area:.2f}, 周长: {perimeter:.2f}, 比例: {aspect_ratio:.2f}")
                                return frame_contour
        
        # 最后的备选方案：检测最大矩形区域
        if frame_contour is None:
            logging.info("尝试检测最大矩形区域作为备选方案")
            _, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w_rect, h_rect = cv2.boundingRect(largest_contour)
                frame_contour = np.array([
                    [[x, y]], [[x + w_rect, y]],
                    [[x + w_rect, y + h_rect]], [[x, y + h_rect]]
                ], dtype=np.int32)
        
        return frame_contour
    
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标点击回调函数，用于手动标记"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.manual_points.append((x, y))
            img = param["img"].copy()
            # 绘制已选点
            for i, (px, py) in enumerate(self.manual_points):
                cv2.circle(img, (px, py), 5, (0, 255, 0), -1)
                cv2.putText(img, str(i+1), (px+10, py), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # 绘制连接线
            if len(self.manual_points) > 1:
                for i in range(len(self.manual_points)-1):
                    cv2.line(img, self.manual_points[i], self.manual_points[i+1], 
                            (0, 255, 0), 2)
            # 如果选了4个点，闭合多边形
            if len(self.manual_points) == 4:
                cv2.line(img, self.manual_points[3], self.manual_points[0], 
                        (0, 255, 0), 2)
                cv2.putText(img, "按Enter确认，按ESC重新选择", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("手动标记画框 (点击4个角点: 左上→右上→右下→左下)", img)
    
    def _manual_selection(self, img):
        """手动选择画框四个角点"""
        self.manual_points = []
        cv2.namedWindow("手动标记画框 (点击4个角点: 左上→右上→右下→左下)")
        cv2.setMouseCallback(
            "手动标记画框 (点击4个角点: 左上→右上→右下→左下)",
            self._mouse_callback,
            {"img": img}
        )
        cv2.imshow("手动标记画框 (点击4个角点: 左上→右上→右下→左下)", img)
        
        while True:
            key = cv2.waitKey(0)
            if key == 13:  # Enter键确认
                if len(self.manual_points) == 4:
                    cv2.destroyAllWindows()
                    return np.array(self.manual_points, dtype=np.int32).reshape(-1, 1, 2)
                else:
                    logging.warning("请选择 exactly 4个点")
            elif key == 27:  # ESC键重新选择
                self.manual_points = []
                cv2.imshow("手动标记画框 (点击4个角点: 左上→右上→右下→左下)", img)
                logging.info("重新选择画框角点")
        
        cv2.destroyAllWindows()
        return None
    
    def _perspective_transform(self, img, contour):
        """透视变换去除画框"""
        # 提取并排序顶点
        points = contour.reshape(4, 2).astype("float32")
        
        # 排序四个顶点（左上、右上、右下、左下）
        s = points.sum(axis=1)
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = points[np.argmin(s)]  # 左上
        rect[2] = points[np.argmax(s)]  # 右下
        
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]  # 右上
        rect[3] = points[np.argmax(diff)]  # 左下
        
        # 计算目标尺寸
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        
        # 目标坐标
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        # 计算透视变换矩阵并应用
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        
        # 轻微裁剪边缘，去除可能的画框残留
        h, w = warped.shape[:2]
        margin = max(5, int(max(w, h) * 0.01))
        if h > 2 * margin and w > 2 * margin:
            warped = warped[margin:h-margin, margin:w-margin]
            
        return warped
    
    def remove_frame(self, image_path, output_path=None, debug=False, manual=False, 
                    preprocess_strategy="auto"):
        """
        移除图像中的画框
        
        参数:
            image_path: 输入图像路径
            output_path: 输出图像路径，None则自动生成
            debug: 是否显示调试信息和中间结果
            manual: 是否使用手动标记模式
            preprocess_strategy: 预处理策略，"auto"自动选择，或指定"low_light", "high_light", "low_contrast", "normal"
        """
        self.debug_mode = debug
        start_time = datetime.now()
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"无法读取图像: {image_path}")
            return None
        
        # 创建输出路径
        if output_path is None:
            img_dir = os.path.dirname(image_path)
            img_name = os.path.basename(image_path)
            name, ext = os.path.splitext(img_name)
            output_path = os.path.join(img_dir, f"{name}_no_frame{ext}")
        
        # 复制原始图像
        original = img.copy()
        h, w = img.shape[:2]
        logging.info(f"处理图像: {image_path} (尺寸: {w}x{h})")
        
        # 手动模式
        if manual:
            logging.info("使用手动标记模式")
            frame_contour = self._manual_selection(img)
            if frame_contour is None:
                logging.error("手动标记失败")
                return None
        else:
            # 自动检测模式
            # 1. 预处理
            processed = self._preprocess_image(img, preprocess_strategy)
            
            # 2. 边缘检测
            edges = self._detect_edges(processed)
            
            # 3. 寻找画框轮廓
            frame_contour = self._find_frame_contour(img, edges)
            
            # 如果未找到轮廓，尝试自动调整参数再试一次
            if frame_contour is None:
                logging.warning("首次检测失败，尝试调整参数再检测一次")
                # 尝试更强的预处理
                processed = self._preprocess_image(img, "low_contrast")  # 强制增强对比度
                edges = self._detect_edges(processed)
                frame_contour = self._find_frame_contour(img, edges)
        
        # 检查是否找到画框
        if frame_contour is None:
            logging.error("无法检测到画框，请尝试手动模式")
            # 保存原始图像作为备选
            cv2.imwrite(output_path, original)
            return output_path
        
        # 显示检测到的轮廓（调试模式）
        if self.debug_mode and not manual:
            temp = img.copy()
            cv2.drawContours(temp, [frame_contour], -1, (0, 255, 0), 2)
            cv2.imshow("检测到的画框轮廓", temp)
            cv2.waitKey(0)
        
        # 执行透视变换
        result = self._perspective_transform(original, frame_contour)
        
        # 保存结果
        cv2.imwrite(output_path, result)
        
        # 显示最终结果（调试模式）
        if self.debug_mode:
            cv2.imshow("移除画框结果", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # 计算处理时间
        elapsed = (datetime.now() - start_time).total_seconds()
        logging.info(f"处理完成，结果保存至: {output_path} (耗时: {elapsed:.2f}秒)")
        
        return output_path
    
    def batch_process(self, input_dir, output_dir=None, debug=False, 
                     auto_fallback_manual=False):
        """
        批量处理目录中的图像
        
        参数:
            input_dir: 输入图像目录
            output_dir: 输出目录，None则创建"no_frame_output"子目录
            debug: 是否显示调试信息
            auto_fallback_manual: 自动检测失败时是否自动切换到手动模式
        """
        # 创建输出目录
        if output_dir is None:
            output_dir = os.path.join(input_dir, "no_frame_output")
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"开始批量处理，输入目录: {input_dir}, 输出目录: {output_dir}")
        
        # 支持的图像格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and
            os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        if not image_files:
            logging.warning(f"在目录 {input_dir} 中未找到图像文件")
            return
        
        # 处理每个图像
        for i, filename in enumerate(image_files, 1):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            logging.info(f"处理图像 {i}/{len(image_files)}: {filename}")
            
            try:
                # 先尝试自动检测
                result = self.remove_frame(
                    input_path, 
                    output_path, 
                    debug=debug, 
                    manual=False
                )
                
                # 如果自动检测失败且启用了手动回退
                if auto_fallback_manual and result is None:
                    logging.info(f"对 {filename} 启用手动模式")
                    self.remove_frame(
                        input_path, 
                        output_path, 
                        debug=debug, 
                        manual=True
                    )
                    
            except Exception as e:
                logging.error(f"处理 {filename} 时出错: {str(e)}")
                continue
        
        logging.info(f"批量处理完成，共处理 {len(image_files)} 个文件，结果保存在 {output_dir}")

if __name__ == "__main__":
    # 创建画框移除器实例
    frame_remover = FrameRemover()
    
    # 示例：处理单个图像
    # frame_remover.remove_frame(
    #     image_path="input.jpg",
    #     output_path="output.jpg",
    #     debug=True,
    #     manual=False  # 自动检测失败时改为True启用手动模式
    # )
    
    # 示例：批量处理目录中的图像
    frame_remover.batch_process(
        input_dir="corrected_photo.jpg",  # 替换为你的图像目录
        output_dir="output.jpg", # 替换为输出目录
        debug=False,
        auto_fallback_manual=True    # 自动检测失败时切换到手动模式
    )
