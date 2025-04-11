import os
import re
import pysrt
import nltk
import stanza
from gtts import gTTS
from googletrans import Translator
from Levenshtein import distance
from typing import List, Dict, Tuple, Optional
import logging
import time
import requests
from requests.exceptions import ConnectionError
import traceback

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LanguageAnalyzer:
    def __init__(self, srt_file: str, target_language: str = 'en', max_retries: int = 3):
        """
        初始化语言分析器
        
        Args:
            srt_file: SRT字幕文件路径
            target_language: 目标语言代码，默认为英语
            max_retries: 模型下载最大重试次数
        """
        self.srt_file = srt_file
        self.target_language = target_language
        self.translator = Translator()
        
        # 下载必要的NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # 初始化Stanza，添加重试机制
        self.nlp = self._initialize_stanza(max_retries)
        
        # 加载字幕
        self.subtitles = self._load_subtitles()
        
    def _initialize_stanza(self, max_retries: int) -> stanza.Pipeline:
        """
        初始化Stanza，包含重试机制
        
        Args:
            max_retries: 最大重试次数
            
        Returns:
            Stanza Pipeline对象
        """
        for attempt in range(max_retries):
            try:
                # 检查模型是否已下载
                if not os.path.exists(os.path.expanduser('~/stanza_resources')):
                    logger.info("正在下载Stanza模型...")
                    stanza.download('en')
                
                return stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')
                
            except (ConnectionError, ConnectionResetError) as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 递增等待时间
                    logger.warning(f"下载模型失败，{wait_time}秒后重试... (尝试 {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error("无法下载Stanza模型，请检查网络连接或手动下载模型")
                    raise
            except Exception as e:
                logger.error(f"初始化Stanza时发生错误: {e}")
                raise
                
    def _load_subtitles(self) -> List[pysrt.SubRipItem]:
        """加载SRT字幕文件"""
        try:
            return pysrt.open(self.srt_file)
        except Exception as e:
            logger.error(f"加载字幕文件失败: {e}")
            raise
            
    def _clean_text(self, text: str) -> str:
        """清理文本，移除特殊字符和多余空格"""
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除特殊字符，保留基本标点
        text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
        # 规范化空格
        text = ' '.join(text.split())
        return text.strip()
        
    def analyze_subtitle(self, subtitle: pysrt.SubRipItem) -> Dict:
        """
        分析单个字幕条目
        
        Args:
            subtitle: 字幕条目
            
        Returns:
            包含分析结果的字典
        """
        try:
            # 清理文本
            clean_text = self._clean_text(subtitle.text)
            
            # 使用Stanza进行语言分析
            doc = self.nlp(clean_text)
            
            # 提取词性标注和依存关系
            pos_tags = []
            dependencies = []
            for sentence in doc.sentences:
                for word in sentence.words:
                    pos_tags.append({
                        'word': word.text,
                        'pos': word.pos,
                        'lemma': word.lemma
                    })
                    dependencies.append({
                        'word': word.text,
                        'head': sentence.words[word.head-1].text if word.head > 0 else 'ROOT',
                        'deprel': word.deprel
                    })
            
            # 翻译文本
            translation = self.translator.translate(clean_text, dest=self.target_language)
            
            return {
                'original_text': clean_text,
                'translation': translation.text,
                'pos_tags': pos_tags,
                'dependencies': dependencies,
                'start_time': str(subtitle.start),
                'end_time': str(subtitle.end)
            }
            
        except Exception as e:
            logger.error(f"分析字幕失败: {e}")
            return {
                'original_text': subtitle.text,
                'error': str(e)
            }
            
    def analyze_all(self) -> List[Dict]:
        """
        分析所有字幕
        
        Returns:
            包含所有字幕分析结果的列表
        """
        results = []
        for subtitle in self.subtitles:
            result = self.analyze_subtitle(subtitle)
            results.append(result)
        return results
        
    def generate_audio(self, text: str, output_file: str) -> None:
        """
        生成音频文件
        
        Args:
            text: 要转换为音频的文本
            output_file: 输出音频文件路径
        """
        try:
            tts = gTTS(text=text, lang=self.target_language)
            tts.save(output_file)
            logger.info(f"音频文件已生成: {output_file}")
        except Exception as e:
            logger.error(f"生成音频失败: {e}")
            raise
            
    def find_similar_phrases(self, phrase: str, threshold: float = 0.8) -> List[Dict]:
        """
        查找相似短语
        
        Args:
            phrase: 要查找的短语
            threshold: 相似度阈值
            
        Returns:
            包含相似短语及其信息的列表
        """
        results = []
        phrase = phrase.lower()
        
        for subtitle in self.subtitles:
            text = subtitle.text.lower()
            # 计算编辑距离
            dist = distance(phrase, text)
            similarity = 1 - (dist / max(len(phrase), len(text)))
            
            if similarity >= threshold:
                results.append({
                    'text': subtitle.text,
                    'similarity': similarity,
                    'start_time': str(subtitle.start),
                    'end_time': str(subtitle.end)
                })
                
        return sorted(results, key=lambda x: x['similarity'], reverse=True)

def main():
    try:
        # 示例用法
        srt_file = "demo.srt"
        analyzer = LanguageAnalyzer(srt_file)
        
        # 分析所有字幕
        results = analyzer.analyze_all()
        
        # 输出分析结果
        for result in results:
            print(f"\n原文: {result['original_text']}")
            print(f"翻译: {result['translation']}")
            print("词性标注:")
            for tag in result['pos_tags']:
                print(f"  {tag['word']}: {tag['pos']} ({tag['lemma']})")
            print("依存关系:")
            for dep in result['dependencies']:
                print(f"  {dep['word']} <-{dep['deprel']}- {dep['head']}")
                
        # 生成音频
        analyzer.generate_audio("Hello, this is a test.", "test.mp3")
        
        # 查找相似短语
        similar = analyzer.find_similar_phrases("test phrase")
        print("\n相似短语:")
        for item in similar:
            print(f"{item['text']} (相似度: {item['similarity']:.2f})")
            
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        print(f"错误: {e}")
        print("\n如果遇到网络问题，请尝试以下解决方案：")
        print("1. 检查网络连接")
        print("2. 使用代理或VPN")
        print("3. 手动下载Stanza模型：")
        print("   - 访问 https://stanfordnlp.github.io/stanza/")
        print("   - 下载所需的模型文件")
        print("   - 将模型文件放在 ~/stanza_resources 目录下")
        
        traceback.print_stack()

if __name__ == "__main__":
    main() 