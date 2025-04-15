import logging
import pysrt
import stanza
import re
import os
import time
import requests
import nltk
import phonetics
from typing import List, Dict, Tuple, Optional, Any
from nltk.corpus import wordnet
from googletrans import Translator
from stanza.pipeline.core import DownloadMethod
from gtts import gTTS
from Levenshtein import distance
from pathlib import Path
from collections import defaultdict
import json
import asyncio
import traceback
from cefrpy import CEFRAnalyzer
import eng_to_ipa as ipa
from reportlab.lib import colors
from reportlab.lib.pagesizes import A5
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from ebooklib import epub

# Configure logging
logging.basicConfig(level=logging.INFO)  # 降低日志级别
logger = logging.getLogger(__name__)

# Initialize translator and other components
translator = Translator()

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')

class LanguageAnalyzer:
    def __init__(self, srt_file: str, target_language: str = 'en', max_retries: int = 3, offline_mode: bool = False):
        """
        初始化语言分析器
        
        Args:
            srt_file: SRT字幕文件路径
            target_language: 目标语言代码，默认为英语
            max_retries: 模型下载最大重试次数
            offline_mode: 是否使用离线模式（不下载模型）
        """
        self.srt_file = srt_file
        self.target_language = target_language
        self.translator = Translator()
        self.offline_mode = offline_mode
        
        # 创建generated目录
        self.output_dir = Path("generated")
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化Stanza，添加重试机制
        self.nlp = self._initialize_stanza(max_retries)
        
        # 初始化CEFR分类器
        self.cefr = CEFRAnalyzer()
        
        # 加载字幕
        self.subtitles = self._load_subtitles()
        
        # 初始化难度分析相关
        self.words = set()  # 存储所有单词
        self.phrases = set()  # 存储所有短语
        self.merged_sentences = []  # 存储合并后的句子
        self.difficult_words = set()
        self.difficult_phrases = set()
        self.difficult_sentences = set()
        
    def _get_stanza_model_path(self) -> Path:
        """获取Stanza模型路径"""
        home = Path.home()
        model_path = home / 'stanza_resources'
        if not model_path.exists():
            model_path = Path('/usr/local/share/stanza_resources')
        if not model_path.exists():
            model_path = Path('/usr/share/stanza_resources')
        return model_path / 'en'
        
    def _check_model_exists(self) -> bool:
        """检查模型是否已下载"""
        model_path = self._get_stanza_model_path()
        if not model_path.exists():
            return False
            
        # 检查必要的模型文件
        required_files = [
            'default.zip',
            'tokenize/combined.pt',
            'pos/combined_charlm.pt',
            'lemma/combined_nocharlm.pt',
            'depparse/combined_charlm.pt'
        ]
        
        for file in required_files:
            if not (model_path / file).exists():
                logging.debug(f"缺少模型文件: {file}")
                return False
                
        return True
        
    def _initialize_stanza(self, max_retries: int) -> stanza.Pipeline:
        """
        初始化Stanza，包含重试机制
        
        Args:
            max_retries: 最大重试次数
            
        Returns:
            Stanza Pipeline对象
        """
        if self.offline_mode:
            if not self._check_model_exists():
                raise RuntimeError("离线模式下未找到Stanza模型，请先下载模型或禁用离线模式")
            logging.info("使用离线模式初始化Stanza")
            return stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse', download_method=None)
            
        for attempt in range(max_retries):
            try:
                # 检查模型是否已下载
                if not self._check_model_exists():
                    logging.info("正在下载Stanza模型...")
                    stanza.download('en', model_dir=str(self._get_stanza_model_path().parent))
                
                return stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse', download_method=DownloadMethod.REUSE_RESOURCES)
                
            except (ConnectionError, ConnectionResetError) as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 递增等待时间
                    logging.warning(f"下载模型失败，{wait_time}秒后重试... (尝试 {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logging.error("无法下载Stanza模型，请检查网络连接或手动下载模型")
                    raise
            except Exception as e:
                logging.error(f"初始化Stanza时发生错误: {e}")
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

    def _merge_sentences(self) -> List[Dict]:
        """
        合并被拆分的句子
        
        Returns:
            合并后的句子列表，每个句子包含文本和时间信息
        """
        merged_sentences = []
        current_sentence = ""
        current_start = None
        current_end = None
        
        for sub in self.subtitles:
            text = self._clean_text(sub.text)
            
            # 如果当前句子为空，开始新的句子
            if not current_sentence:
                current_sentence = text
                current_start = sub.start
                current_end = sub.end
            else:
                # 检查是否是同一个句子的继续
                # 放宽句子合并条件
                if (not text[0].isupper() or 
                    text.startswith('"') or 
                    text.startswith("'") or
                    text.startswith('(') or
                    text.startswith('[') or
                    text.startswith('{') or
                    len(current_sentence) < 5 or  # 如果当前句子太短，继续合并
                    text.startswith('and ') or    # 以and开头的句子继续合并
                    text.startswith('but ') or    # 以but开头的句子继续合并
                    text.startswith('or ') or     # 以or开头的句子继续合并
                    text.startswith('so ') or     # 以so开头的句子继续合并
                    text.startswith('for ') or    # 以for开头的句子继续合并
                    text.startswith('nor ') or    # 以nor开头的句子继续合并
                    text.startswith('yet ')):     # 以yet开头的句子继续合并
                    current_sentence += " " + text
                    current_end = sub.end
                else:
                    # 保存当前句子并开始新的句子
                    if len(current_sentence.split()) <= 50:  # 放宽句子长度限制
                        merged_sentences.append({
                            'text': current_sentence,
                            'start': current_start,
                            'end': current_end
                        })
                    current_sentence = text
                    current_start = sub.start
                    current_end = sub.end
        
        # 添加最后一个句子
        if current_sentence and len(current_sentence.split()) <= 50:
            merged_sentences.append({
                'text': current_sentence,
                'start': current_start,
                'end': current_end
            })
            
        return merged_sentences
        
    def _extract_phrases(self, text: str) -> List[str]:
        """
        从文本中提取有意义的短语
        
        Args:
            text: 要分析的文本
            
        Returns:
            提取出的短语列表
        """
        try:
            # 使用Stanza进行依存分析
            doc = self.nlp(text)
            phrases = []
            
            for sentence in doc.sentences:
                # 提取名词短语
                noun_phrases = []
                current_phrase = []
                
                for word in sentence.words:
                    # 如果是名词、形容词、限定词、代词或数词，添加到当前短语
                    if word.pos in ['NOUN', 'PROPN', 'ADJ', 'DET', 'PRON', 'NUM']:
                        current_phrase.append(word.text)
                    # 如果是介词，开始新的短语
                    elif word.pos == 'ADP' and current_phrase:
                        noun_phrases.append(' '.join(current_phrase))
                        current_phrase = []
                    # 如果是其他词性且当前短语不为空，保存当前短语
                    elif current_phrase:
                        noun_phrases.append(' '.join(current_phrase))
                        current_phrase = []
                
                # 添加最后一个短语
                if current_phrase:
                    noun_phrases.append(' '.join(current_phrase))
                
                # 提取动词短语
                verb_phrases = []
                current_phrase = []
                
                for word in sentence.words:
                    # 如果是动词、助动词、副词、介词或连词，添加到当前短语
                    if word.pos in ['VERB', 'AUX', 'ADV', 'ADP', 'CCONJ', 'SCONJ']:
                        current_phrase.append(word.text)
                    # 如果是其他词性且当前短语不为空，保存当前短语
                    elif current_phrase:
                        verb_phrases.append(' '.join(current_phrase))
                        current_phrase = []
                
                # 添加最后一个短语
                if current_phrase:
                    verb_phrases.append(' '.join(current_phrase))
                
                # 合并所有短语
                phrases.extend(noun_phrases)
                phrases.extend(verb_phrases)
            
            # 过滤掉不合理的短语
            valid_phrases = []
            for phrase in phrases:
                words = phrase.split()
                # 短语长度在2-6个词之间
                if 2 <= len(words) <= 6:
                    # 检查短语是否合理
                    if self._is_valid_phrase(phrase):
                        valid_phrases.append(phrase)
            
            return valid_phrases
            
        except Exception as e:
            logger.error(f"提取短语失败: {e}")
            return []
            
    def _is_valid_phrase(self, phrase: str) -> bool:
        """
        判断短语是否合理
        
        Args:
            phrase: 要判断的短语
            
        Returns:
            是否合理
        """
        # 检查是否包含重复词
        words = phrase.split()
        if len(words) != len(set(words)):
            return False
            
        # 检查是否包含不合理的组合
        invalid_patterns = [
            r'\b\w+\s+\w+\s+\w+\s+\w+\s+\w+\b',  # 太长的短语
            r'\b[a-z]\s+[a-z]\b',                  # 单个字母的组合
            r'\b\d+\s+\w+\b',                      # 数字和词的组合
            r'\b\w+\s+\d+\b'                       # 词和数字的组合
        ]
        
        return not any(re.search(pattern, phrase.lower()) for pattern in invalid_patterns)
        
    def _get_word_level(self, word: str) -> str:
        """使用cefrpy库确定单词的CEFR等级"""
        try:
            # 清理单词
            word = word.lower().strip('.,!?;:\'\"')
            if not word or len(word) <= 1:
                return 'A1-A2'
                
            # 使用CEFR分类器获取等级
            level = self.cefr.get_average_word_level_float(word)
            if not level:
                return 'A1-A2'
                
            # 将数字等级转换为字母等级
            level_map = {
                1: 'A1',
                2: 'A2',
                3: 'B1',
                4: 'B2',
                5: 'C1',
                6: 'C2'
            }
            return level_map.get(level, 'A1-A2')
            
        except Exception as e:
            logging.error(f"获取单词 {word} 的CEFR等级时出错: {str(e)}")
            return 'A1-A2'

    def _is_difficult_word(self, word: str) -> bool:
        """判断一个单词是否为难词（B1及以上级别）"""
        try:
            # 获取单词的CEFR等级
            level = self._get_word_level(word)
            
            # 如果等级是B1或以上，认为是难词
            if level in ['B1', 'B2', 'C1', 'C2']:
                logging.debug(f"单词 {word} 的难度级别为 {level}")
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"判断单词 {word} 难度时出错: {str(e)}")
            return False

    async def get_word_info(self, word: str) -> dict:
        """
        获取单词的详细信息，包括音标、中文释义和英文定义
        """
        try:
            # 初始化结果字典
            result = {
                'phonetic': '',
                'chinese': '',
                'definitions': []
            }
            
            # 获取音标
            try:
                # 使用eng-to-ipa获取音标
                phonetic = ipa.convert(word)
                if phonetic:
                    result['phonetic'] = phonetic
            except Exception as e:
                logging.warning(f"获取单词 {word} 音标时出错: {str(e)}")
            
            # 获取英文定义和中文释义
            try:
                # 使用wordnet获取定义
                synsets = wordnet.synsets(word)
                if synsets:
                    # 获取英文定义
                    for synset in synsets[:3]:  # 最多取3个定义
                        result['definitions'].append(synset.definition())
                    
                    # 获取中文释义（使用wordnet的lemma名称作为简单翻译）
                    lemmas = synsets[0].lemmas()
                    if lemmas:
                        result['chinese'] = lemmas[0].name()
            except Exception as e:
                logging.warning(f"获取单词 {word} 释义时出错: {str(e)}")
            
            return result
        except Exception as e:
            logging.error(f"获取单词 {word} 信息时出错: {str(e)}")
            return {'phonetic': '', 'chinese': '', 'definitions': []}

    def _convert_to_pdf(self, txt_file: Path) -> Path:
        """
        将TXT文件转换为适合手机查看的PDF格式
        
        Args:
            txt_file: TXT文件路径
            
        Returns:
            PDF文件路径
        """
        try:
            pdf_file = txt_file.with_suffix('.pdf')
            
            # 创建PDF文档
            doc = SimpleDocTemplate(
                str(pdf_file),
                pagesize=A5,  # 使用A5尺寸，更适合手机
                rightMargin=30,
                leftMargin=30,
                topMargin=30,
                bottomMargin=30
            )
            
            # 创建样式
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=14,
                spaceAfter=20
            )
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=12,
                leading=16
            )
            
            # 读取TXT内容
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 转换内容为PDF元素
            elements = []
            
            # 添加标题
            title = txt_file.stem
            elements.append(Paragraph(title, title_style))
            elements.append(Spacer(1, 12))
            
            # 添加正文内容
            for line in content.split('\n'):
                if line.strip():
                    p = Paragraph(line, normal_style)
                    elements.append(p)
                    elements.append(Spacer(1, 6))
            
            # 生成PDF
            doc.build(elements)
            
            logging.info(f"已生成PDF文件：{pdf_file}")
            return pdf_file
            
        except Exception as e:
            logging.error(f"转换PDF时出错: {str(e)}")
            return None
            
    def _convert_to_epub(self, txt_file: Path) -> Path:
        """
        将TXT文件转换为EPUB格式
        
        Args:
            txt_file: TXT文件路径
            
        Returns:
            EPUB文件路径
        """
        try:
            epub_file = txt_file.with_suffix('.epub')
            
            # 创建EPUB书籍
            book = epub.EpubBook()
            
            # 设置元数据
            book.set_identifier(str(txt_file.stem))
            book.set_title(txt_file.stem)
            book.set_language('en')
            
            # 读取TXT内容
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 创建章节
            c1 = epub.EpubHtml(title=txt_file.stem,
                              file_name='content.xhtml',
                              lang='en')
            
            # 将内容转换为HTML格式
            html_content = ['<h1>{}</h1>'.format(txt_file.stem)]
            for line in content.split('\n'):
                if line.strip():
                    html_content.append('<p>{}</p>'.format(line))
            
            c1.content = '\n'.join(html_content)
            
            # 添加章节
            book.add_item(c1)
            
            # 创建目录
            book.toc = [(epub.Section('Main'), [c1])]
            
            # 添加默认NCX和Nav文件
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())
            
            # 定义阅读顺序
            book.spine = ['nav', c1]
            
            # 生成EPUB
            epub.write_epub(str(epub_file), book, {})
            
            logging.info(f"已生成EPUB文件：{epub_file}")
            return epub_file
            
        except Exception as e:
            logging.error(f"转换EPUB时出错: {str(e)}")
            return None

    def _write_results(self, file_prefix: str, difficult_words: set, difficult_phrases: list, difficult_sentences: list):
        """
        将分析结果写入文件
        """
        try:
            # 写入难词信息
            output_path = self.output_dir / f"{file_prefix}_words.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                for word in sorted(difficult_words):
                    word_info = asyncio.run(self.get_word_info(word[0]))
                    level = self._get_word_level(word[0])
                    f.write(f"{word[0]} (CEFR: {level})\n")
                    if word_info['phonetic']:
                        f.write(f"  音标: /{word_info['phonetic']}/\n")
                    if word_info['chinese']:
                        f.write(f"  中文释义: {word_info['chinese']}\n")
                    if word_info['definitions']:
                        f.write("  英文定义:\n")
                        for i, definition in enumerate(word_info['definitions'], 1):
                            f.write(f"    {i}. {definition}\n")
                    f.write("\n")
            
            # 写入难词组信息
            output_path = self.output_dir / f"{file_prefix}_phrases.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                for phrase in difficult_phrases:
                    f.write(f"{phrase}\n")
            
            # 写入难句信息
            output_path = self.output_dir / f"{file_prefix}_sentences.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                for sentence in difficult_sentences:
                    f.write(f"{sentence}\n")
            
            # 转换为PDF和EPUB格式
            txt_files = [
                self.output_dir / f"{file_prefix}_words.txt",
                self.output_dir / f"{file_prefix}_phrases.txt",
                self.output_dir / f"{file_prefix}_sentences.txt"
            ]
            
            for txt_file in txt_files:
                if txt_file.exists():
                    self._convert_to_pdf(txt_file)
                    self._convert_to_epub(txt_file)
            
            logging.info(f"分析结果已写入文件并转换为PDF和EPUB格式：\n"
                        f"{self.output_dir}/{file_prefix}_words.txt/.pdf/.epub\n"
                        f"{self.output_dir}/{file_prefix}_phrases.txt/.pdf/.epub\n"
                        f"{self.output_dir}/{file_prefix}_sentences.txt/.pdf/.epub")
        except Exception as e:
            logging.error(f"写入结果时出错: {str(e)}")
            traceback.print_exc()

    def _is_difficult_phrase(self, phrase):
        """判断短语是否困难"""
        words = phrase.lower().split()
        difficult_word_count = sum(1 for word in words if self._is_difficult_word(word))
        return difficult_word_count >= len(words) / 2

    def analyze_difficulty(self, file_prefix):
        """分析文本难度并写入结果"""
        try:
            # 获取所有单词和短语
            self.merged_sentences = self._merge_sentences()
            logging.info(f"合并后的句子数量: {len(self.merged_sentences)}")
            
            # 处理每个句子
            for sentence in self.merged_sentences:
                text = sentence['text']
                # 提取单词
                words = text.lower().split()
                self.words.update(words)
                
                # 提取短语
                phrases = self._extract_phrases(text)
                self.phrases.update(phrases)
            
            logging.info(f"提取到的单词数量: {len(self.words)}")
            logging.info(f"提取到的短语数量: {len(self.phrases)}")

            # 分析单词难度
            difficult_words = []
            for word in self.words:
                if self._is_difficult_word(word):
                    word_info = asyncio.run(self.get_word_info(word))
                    difficult_words.append((word, word_info))
                    logging.debug(f"找到难词: {word}")

            logging.info(f"找到的难词数量: {len(difficult_words)}")

            # 分析短语难度
            difficult_phrases = [phrase for phrase in self.phrases if self._is_difficult_phrase(phrase)]
            logging.info(f"找到的难词组数量: {len(difficult_phrases)}")

            # 分析句子难度
            difficult_sentences = []
            for sentence in self.merged_sentences:
                text = sentence['text']
                words = text.split()
                difficult_word_count = sum(1 for word in words if self._is_difficult_word(word))
                if difficult_word_count >= len(words) / 3:  # 如果超过1/3的单词是困难的
                    difficult_sentences.append(text)
                    logging.debug(f"找到难句: {text}")

            logging.info(f"找到的难句数量: {len(difficult_sentences)}")

            # 写入结果
            self._write_results(file_prefix, difficult_words, difficult_phrases, difficult_sentences)
            logging.info(f"分析完成，结果已写入文件")

        except Exception as e:
            logging.error(f"分析难度时出错: {str(e)}")
            raise

def main():
    try:
        # 示例用法
        # srt_file = "demo.srt"
        srt_file_dict = {
            # "CNN This Morning-20250408":"CNN This Morning-20250408-Trump Threatens China-WMHY5057419108.srt",
            # "JoeRogan-2294":"JoeRogan-2294-GLT1061251245-2294 - Dr. Suzanne Humphries.srt",
            "demo":"demo.srt",
        }
        
        for file_prefix, srt_file in srt_file_dict.items():
            # 检查是否使用离线模式
            offline_mode = os.environ.get('STANZA_OFFLINE', 'false').lower() == 'true'
            analyzer = LanguageAnalyzer(srt_file, offline_mode=offline_mode)
            
            # 分析难度
            analyzer.analyze_difficulty(file_prefix)
        
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        print(f"错误: {e}")
        print("\n如果遇到网络问题，请尝试以下解决方案：")
        print("1. 检查网络连接")
        print("2. 使用代理或VPN")
        print("3. 使用离线模式（如果已下载模型）：")
        print("   export STANZA_OFFLINE=true")
        print("4. 手动下载Stanza模型：")
        print("   - 访问 https://stanfordnlp.github.io/stanza/")
        print("   - 下载所需的模型文件")
        print("   - 将模型文件放在 ~/stanza_resources/en 目录下")
        
        traceback.print_exc()

if __name__ == "__main__":
    main() 