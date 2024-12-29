"""
@File    :   ai_api.py
@Time    :   2024/12/29
@Author  :   Galaxy9572
@Version :   1.0.0
@Contact :   ljy957266@gmail.com
@License :   (C)Copyright Galaxy9572
"""
import json
import time
import base64
import logging
from typing import Tuple, Dict, Any, Union

from zhipuai import ZhipuAI
import config

prompt = """
# 任务描述

提取文本中的名片信息，按以下JSON格式输出，输出必须是合法的JSON格式,不含其他文本

{
     "name": "中文姓名",
     "enName": "英文姓名", 
     "phoneNumbers": ["手机号码"],
     "telephoneNumbers": ["电话号码"],
     "faxNumbers": ["传真号码"],
     "emails": ["邮箱"],
     "companies": [{
          "companyName": "公司中文名",
          "companyEnName": "公司英文名",
          "address": "公司中文地址", 
          "enAddress": "公司英文地址",
          "job": [{
               "name": "中文职位",
               "enName": "英文职位"
          }]
     }],
     "website": ["网址"],
     "extra": "其他信息"
}

# 字段填充规则

## 语言识别规则

包含中文字符的文本仅填充到中文字段
仅包含英文字母的文本仅填充到英文字段
严禁将英文内容翻译后填充到中文字段
严禁将中文内容翻译后填充到英文字段

## 姓名

name: 仅填充中文姓名
enName: 仅填充英文姓名

## 公司与职位

companyName: 仅填充中文公司名
companyEnName: 仅填充英文公司名
address: 仅填充中文地址
enAddress: 仅填充英文地址
companies.job.name: 仅填充中文职位名称
companies.job.enName: 仅填充英文职位名称

## 联系方式格式化

phoneNumbers: 仅填充手机号码
telephoneNumbers: 仅填充电话号码
faxNumbers: 仅填充传真号码
号码格式: (区号)号码
根据上下文判断号码类型，填充到对应字段

## 通用规则

识别不到信息时保持字段为空
数组类型字段支持多个值
extra字段仅存储无法归类的信息
"""

client = ZhipuAI(api_key=config.Config.ZHIPUAI_API_KEY)  # 请填写您自己的APIKey

# 设置日志
logger = logging.getLogger(__name__)

class AIRecognitionError(Exception):
    """AI识别过程中的自定义异常"""
    pass


def text_recognize(text: str) -> Tuple[Dict[str, Any], float]:
    """
    使用AI模型识别和提取文本信息

    参数:
        text (str): 待分析的输入文本

    返回:
        Tuple[Dict[str, Any], float]: (识别结果, 处理时间)的元组

    异常:
        AIRecognitionError: 文本识别过程中出现错误时抛出
    """
    start_time = time.time()

    try:
        # 验证输入
        if not text or not isinstance(text, str):
            raise ValueError("无效的输入文本")

        # 调用AI接口
        response = client.chat.completions.create(
            model=config.Config.ZHIPUAI_LANGUAGE_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            response_format={
                'type': 'json_object'
            }
        )

        # 处理响应
        if not response or not response.choices:
            raise AIRecognitionError("AI服务返回空响应")

        content = response.choices[0].message.content
        if not content:
            raise AIRecognitionError("AI响应内容为空")

        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            raise AIRecognitionError(f"AI响应JSON解析失败: {str(e)}")

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"文本识别失败，耗时{total_time:.2f}秒: {str(e)}")
        raise AIRecognitionError(f"文本识别失败: {str(e)}")

    total_time = time.time() - start_time
    logger.info(f"文本识别完成，耗时{total_time:.2f}秒")

    return result, total_time


def image_recognize(img_path: str) -> Tuple[Dict[str, Any], float]:
    """
    使用AI视觉模型识别和提取图片信息

    参数:
        img_path (str): 图片文件路径

    返回:
        Tuple[Dict[str, Any], float]: (识别结果, 处理时间)的元组

    异常:
        AIRecognitionError: 图片识别过程中出现错误时抛出
    """
    start_time = time.time()

    try:
        # 验证输入
        if not img_path or not isinstance(img_path, str):
            raise ValueError("无效的图片路径")

        # 读取并编码图片
        try:
            with open(img_path, 'rb') as img_file:
                img_base = base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            raise AIRecognitionError(f"图片读取或编码失败: {str(e)}")

        # 调用AI接口
        response = client.chat.completions.create(
            model=config.Config.ZHIPUAI_VISION_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": img_base}}]}
            ],
            response_format={
                'type': 'json_object'
            }
        )

        # 处理响应
        if not response or not response.choices:
            raise AIRecognitionError("AI视觉服务返回空响应")

        content = response.choices[0].message.content
        if not content:
            raise AIRecognitionError("AI视觉响应内容为空")

        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            raise AIRecognitionError(f"AI视觉响应JSON解析失败: {str(e)}")

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"图片识别失败，耗时{total_time:.2f}秒: {str(e)}")
        raise AIRecognitionError(f"图片识别失败: {str(e)}")

    total_time = time.time() - start_time
    logger.info(f"图片识别完成，耗时{total_time:.2f}秒")

    return result, total_time