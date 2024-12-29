# config.py

class Config:
    # 服务器配置
    HOST = '0.0.0.0'
    PORT = 8080
    DEBUG = False

    # 文件上传配置
    UPLOAD_FOLDER = 'test_data'
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 限制上传文件大小为20MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

    # OCR配置
    OCR_THREAD_WORKERS = 3
    OCR_CLUSTER_EPS = 10
    OCR_MIN_SAMPLES = 1

    # 智谱AI配置
    ZHIPUAI_API_KEY = '你自己的API Key'
    ZHIPUAI_LANGUAGE_MODEL = 'glm-4-flash'
    ZHIPUAI_VISION_MODEL = 'glm-4v-plus'