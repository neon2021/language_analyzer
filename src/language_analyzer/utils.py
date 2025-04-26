import os
import stanza
import logging
from typing import Optional
from pathlib import Path
import re
import time

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    清理文本，移除HTML标签、特殊字符和多余空格
    
    Args:
        text: 要清理的文本
        
    Returns:
        清理后的文本
    """
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除特殊字符，保留基本标点
    text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
    # 移除逗号和感叹号
    text = text.replace(',', '').replace('!', '')
    # 规范化空格
    text = ' '.join(text.split())
    return text.strip()

def get_stanza_model_path() -> Optional[Path]:
    """
    获取Stanza模型路径
    
    Returns:
        Stanza模型路径，如果不存在则返回None
    """
    model_dir = Path.home() / '.stanza'
    if not model_dir.exists():
        return None
    return model_dir

def check_model_exists(lang: str = 'en') -> bool:
    """
    检查Stanza模型是否存在
    
    Args:
        lang: 语言代码
        
    Returns:
        模型是否存在
    """
    model_path = get_stanza_model_path()
    if not model_path:
        return False
    return (model_path / lang).exists()

def initialize_stanza(max_retries: int = 3, offline_mode: bool = False) -> stanza.Pipeline:
    """
    初始化Stanza NLP管道
    
    Args:
        max_retries: 最大重试次数
        offline_mode: 是否使用离线模式
        
    Returns:
        Stanza NLP管道
    """
    if offline_mode:
        if not check_model_exists():
            raise RuntimeError("离线模式下需要预先下载模型")
        return stanza.Pipeline(lang='en', download_method=None)
    
    for attempt in range(max_retries):
        try:
            return stanza.Pipeline(lang='en')
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"初始化Stanza失败: {e}")
                raise
            logger.warning(f"初始化Stanza失败，重试中... ({attempt + 1}/{max_retries})")
            time.sleep(2 ** attempt)  # 指数退避 

def ensure_dir_exists(directory: Path) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 要确保存在的目录路径
        
    Returns:
        创建的目录路径
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    except Exception as e:
        logger.error(f"创建目录失败: {e}")
        raise 