# AI信息提取工具

一个基于 Python 的智能信息提取工具，结合了 PaddleOCR 的文字识别能力和AI的信息提取能力，可以准确地从图片中提取结构化信息。

> ## 1、功能特点

- 支持图片中的信息提取
- 智能文本识别和分组
- 多线程处理提升性能
- 支持中英文混合识别
- 结构化 JSON 输出
- REST API 接口便于集成

> ## 2、环境要求

- Python 3.8+
- PaddleOCR
- Flask
- NumPy
- 智谱AI的API Key

> ## 3、安装步骤

### 3.1、安装 PaddleOCR

#### 3.1.1、安装步骤

> 参考：[PaddleOCR 文档](https://paddlepaddle.github.io/PaddleOCR/latest/quick_start.html)

#### 3.1.2、PaddleOCR支持的语言

>参考：[多语言模型 - PaddleOCR 文档](https://paddlepaddle.github.io/PaddleOCR/latest/ppocr/blog/multi_languages.html#5)

### 3.2、克隆项目

```bash
git clone https://github.com/galaxy9572/ai_extractor.git
cd ai_extractor
```

### 3.3、创建并激活虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Windows下使用: venv\Scripts\activate
```

### 3.4、安装依赖

```bash
pip install -r requirements.txt
```

> ## 4、配置说明

### 4.1. 智谱AI API 配置

- 在[智谱AI开放平台](https://open.zhipuai.cn/)注册账号
- 在控制台生成 API key
- 在 `Config.py` 中替换 API key

```python
ZHIPUAI_API_KEY = '你的API密钥'
```

### 4.2、应用配置

在 `config.py` 中可以自定义：
- 服务器主机和端口
- 上传文件目录
- 允许的文件类型
- OCR 处理参数

在 `ai_api.py` 中修改prompt以实现你自己的提取需求

> ## 5、使用方法

### 5.1. 启动服务器

```bash
python app.py
```

### 5.2、发送请求

API详见本文中 API 接口说明部分

API 将返回 JSON 格式的响应，包含：
- 各阶段处理时间
- 结构化的提取信息

> ## 6、API 接口说明

### 6.1、使用OCR识别分析图片并提取信息

```text
POST /api/ocr/extract
```

请求：

```text
方法：POST
Content-Type: multipart/form-data
请求体：file（图片文件）
```

响应示例：

```json
{
    "success": true,
    "data": {
        "ocr_time": "1.23",
        "text_recognize_time": "0.45",
        "total_time": "1.68",
        "result": {
            // ...
        }
    }
}
```

```json
{
    "success": false,
    "error": "错误信息"
}
```

### 6.2、使用AI视觉分析来处理名片图片并提取信息

```text
POST /api/ai-vision/extract
```

请求：

```text
方法：POST
Content-Type: multipart/form-data
请求体：file（图片文件）
```

响应示例：

```json
{
    "success": true,
    "data": {
        "total_time": "2.34",
        "result": {
            "name": "张三",
            "enName": "Zhang San",
            // 与上述OCR接口相同的结构
        }
    }
}
```

```json
{
    "success": false,
    "error": "错误信息"
}
```

### 6.3、健康检查接口

```text
GET /api/health
```

响应示例：

```json
{
    "success": true,
    "data": {
        "status": "OK"
    }
}
```

错误响应：

```json
{
    "success": false,
    "error": "具体错误信息"
}
```

注意事项：

上传的文件会在处理完成后自动清理

> ## 7、项目结构

```
ai_extractor/
├── app.py              # Flask 主应用
├── ai_api.py           # AI API 集成
├── config.py           # 配置设置
├── requirements.txt    # 项目依赖
├── test_data/         # 上传文件临时存储
```

> ## 8、技术实现

本应用采用多阶段处理流程：

1. **OCR 处理**：使用 PaddleOCR 检测和识别图片中的文字
2. **文本分组**：使用 DBSCAN 聚类算法对文本行进行分组
3. **信息提取**：利用智谱AI的 API 进行结构化信息提取
4. **结果合并**：综合不同识别方法的结果

> ## 9、参与贡献

欢迎提交 Pull Request 来改进项目！

> ## 10、开源协议

本项目采用 MIT 协议 - 详见 LICENSE 文件

> ## 11、致谢

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 提供 OCR 能力
- [智谱AI](https://open.zhipuai.cn/) 提供 AI 信息提取能力

> ## 12、问题反馈

如有问题或建议，请在 GitHub 仓库提交 Issue。