from flask import Flask, request, jsonify, Response
from werkzeug.utils import secure_filename
from paddleocr import PaddleOCR
import numpy as np
from sklearn.cluster import DBSCAN
from math import atan2, degrees
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import os
from functools import wraps
from typing import Tuple, List, Dict, Any, Optional

from config import Config
from ai_api import text_recognize, image_recognize


def setup_logger() -> logging.Logger:
    """配置并创建日志记录器"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(console_handler)

    return logging.getLogger(__name__)


# 创建Flask应用和日志记录器
app = Flask(__name__)
logger = setup_logger()

# 确保上传目录存在
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)


def init_ocr_engine() -> PaddleOCR:
    """初始化OCR引擎"""
    return PaddleOCR(
        det_model_dir='ch_PP-OCRv4_det_infer',
        rec_model_dir='ch_PP-OCRv4_rec_infer',
        use_angle_cls=True,
        lang='ch',
        det_db_thresh=0.3,
        det_db_box_thresh=0.6,
        det_db_score_mode='slow',
        rec_batch_num=6,
        rec_algorithm='SVTR_LCNet',
        drop_score=0.5,
        cpu_threads=os.cpu_count()
    )


class TextLineRecognizer:
    """文本行识别器：使用多线程处理OCR结果并智能合并同行文本"""

    def __init__(self, eps: float = 10, min_samples: int = 1, max_workers: int = 3):
        self.eps = eps
        self.min_samples = min_samples
        self.base_char_width = 10
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logger
        self.ocr = init_ocr_engine()

    def calculate_rotation_angle(self, position: List[Tuple[float, float]]) -> float:
        """计算文本的旋转角度"""
        try:
            x1, y1 = position[0]
            x2, y2 = position[1]
            return degrees(atan2(y2 - y1, x2 - x1))
        except Exception as e:
            self.logger.error(f"计算旋转角度时出错: {str(e)}")
            return 0

    def normalize_coordinates(self, position: List[Tuple[float, float]], angle: float) -> np.ndarray:
        """将倾斜的坐标归一化到水平方向"""
        try:
            coords = np.array(position)
            theta = np.radians(-angle)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            return coords.dot(rotation_matrix)
        except Exception as e:
            self.logger.error(f"归一化坐标时出错: {str(e)}")
            return np.array(position)

    def process_text_block(self, line: Tuple[List[Tuple[float, float]], Tuple[str, float]]) -> Optional[Dict[str, Any]]:
        """处理单个文本块"""
        try:
            text, confidence = line[1]
            if confidence <= 0.5:
                return None

            position = line[0]
            angle = self.calculate_rotation_angle(position)
            normalized_coords = self.normalize_coordinates(position, angle)
            center_x = normalized_coords[:, 0].mean()
            center_y = normalized_coords[:, 1].mean()

            return {
                'text': text,
                'confidence': confidence,
                'position': position,
                'angle': angle,
                'normalized_x': center_x,
                'normalized_y': center_y
            }
        except Exception as e:
            self.logger.error(f"处理文本块时出错: {str(e)}")
            return None

    def calculate_text_spacing(self, element1: Dict[str, Any], element2: Dict[str, Any]) -> int:
        """计算两个文本块之间应插入的空格数量"""
        try:
            x_distance = element2['normalized_x'] - element1['normalized_x']
            if x_distance > self.base_char_width * 1.5:
                space_count = max(1, int(x_distance / self.base_char_width))
                return min(space_count, 3)
            return 1
        except Exception as e:
            self.logger.error(f"计算文本间距时出错: {str(e)}")
            return 1

    def cluster_text_lines(self, text_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用DBSCAN算法对文本元素进行聚类"""
        if not text_elements:
            return []

        try:
            points = np.array([[element['normalized_y']] for element in text_elements])
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)

            for idx, label in enumerate(clustering.labels_):
                text_elements[idx]['line_cluster'] = label
            return text_elements
        except Exception as e:
            self.logger.error(f"聚类文本行时出错: {str(e)}")
            return text_elements

    def process_image(self, image_path: str) -> Tuple[List[Dict[str, Any]], str]:
        """处理图像并提取文本信息"""
        try:
            total_start_time = time.time()

            # OCR识别
            ocr_start_time = time.time()
            result = self.ocr.ocr(image_path, cls=True)
            ocr_time = time.time() - ocr_start_time
            self.logger.info(f"OCR识别耗时: {ocr_time:.2f}秒")

            # 并行处理文本块
            process_start_time = time.time()
            futures = [
                self.executor.submit(self.process_text_block, line)
                for res in result
                for line in res
            ]
            text_elements = [element for future in futures if (element := future.result())]

            process_time = time.time() - process_start_time
            self.logger.info(f"文本块处理耗时: {process_time:.2f}秒")

            # 文本行聚类和合并
            if not text_elements:
                return [], ''

            cluster_start_time = time.time()
            text_elements = self.cluster_text_lines(text_elements)

            # 按聚类结果分组并合并文本
            grouped_lines = {}
            for element in text_elements:
                cluster = element['line_cluster']
                grouped_lines.setdefault(cluster, []).append(element)

            merged_lines = []
            for cluster in sorted(grouped_lines.keys()):
                line_elements = sorted(grouped_lines[cluster], key=lambda x: x['normalized_x'])
                line_text = line_elements[0]['text']

                for i in range(1, len(line_elements)):
                    space_count = self.calculate_text_spacing(line_elements[i - 1], line_elements[i])
                    line_text += ' ' * space_count + line_elements[i]['text']

                merged_lines.append(line_text)

            cluster_time = time.time() - cluster_start_time
            self.logger.info(f"聚类合并耗时: {cluster_time:.2f}秒")
            self.logger.info(f"总处理耗时: {time.time() - total_start_time:.2f}秒")

            return text_elements, '\n'.join(merged_lines)

        except Exception as e:
            self.logger.error(f"处理图像时出错: {str(e)}")
            return [], ''

    def __del__(self):
        """确保在对象销毁时关闭线程池"""
        self.executor.shutdown(wait=True)


# 通用工具函数
def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def handle_file_upload(request) -> Tuple[str, int]:
    """处理文件上传的通用函数"""
    if 'file' not in request.files:
        return '没有文件上传', 400

    file = request.files['file']
    if file.filename == '':
        return '没有选择文件', 400

    if not allowed_file(file.filename):
        return '不支持的文件类型', 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
    file.save(filepath)
    return filepath, 200


def cleanup_file(filepath: str):
    """清理临时文件"""
    try:
        os.remove(filepath)
    except Exception as e:
        logger.warning(f"删除临时文件失败: {str(e)}")


def api_response(success: bool, data: Any = None, error: str = None) -> tuple[Response, int]:
    """统一的API响应格式"""
    response = {'success': success}
    if success and data is not None:
        response['data'] = data
    if not success and error is not None:
        response['error'] = error
    return jsonify(response), 200 if success else 500


def error_handler(func):
    """错误处理装饰器"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"API错误: {str(e)}")
            return api_response(False, error=str(e))

    return wrapper


# 创建全局识别器实例
recognizer = TextLineRecognizer(
    eps=Config.OCR_CLUSTER_EPS,
    min_samples=Config.OCR_MIN_SAMPLES,
    max_workers=Config.OCR_THREAD_WORKERS
)


# API路由
@app.route('/api/ocr/extract', methods=['POST'])
@error_handler
def ocr_extract():
    """OCR文本提取接口"""
    filepath, status = handle_file_upload(request)
    if status != 200:
        return api_response(False, error=filepath)

    start_time = time.time()
    text_elements, merged_text = recognizer.process_image(filepath)
    ocr_time = time.time() - start_time

    text_result, text_recognize_time = text_recognize(merged_text)
    total_time = ocr_time + text_recognize_time

    cleanup_file(filepath)

    return api_response(True, {
        'ocr_time': f"{ocr_time:.2f}",
        'text_recognize_time': f"{text_recognize_time:.2f}",
        'total_time': f"{total_time:.2f}",
        'result': text_result
    })


@app.route('/api/ai-vision/extract', methods=['POST'])
@error_handler
def ai_vision_extract():
    """AI视觉分析接口"""
    filepath, status = handle_file_upload(request)
    if status != 200:
        return api_response(False, error=filepath)

    image_result, image_recognize_time = image_recognize(filepath)
    cleanup_file(filepath)

    return api_response(True, {
        'total_time': f"{image_recognize_time:.2f}",
        'result': image_result
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return api_response(True, {'status': 'OK'})


if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)