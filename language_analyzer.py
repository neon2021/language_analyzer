import os
import re
import pysrt
import nltk
import stanza
from stanza.pipeline.core import DownloadMethod
from gtts import gTTS
from googletrans import Translator
from Levenshtein import distance
from typing import List, Dict, Tuple, Optional
import logging
import time
import requests
from requests.exceptions import ConnectionError
import traceback
from pathlib import Path
from collections import defaultdict
from cefrpy import CEFRAnalyzer

cefr_analyzer = CEFRAnalyzer()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
        # 下载必要的NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # 初始化Stanza，添加重试机制
        self.nlp = self._initialize_stanza(max_retries)
        
        # 加载字幕
        self.subtitles = self._load_subtitles()
        
        # 初始化难度分析相关
        self.difficult_words = set()
        self.difficult_phrases = set()
        self.difficult_sentences = set()
        
    def _get_stanza_model_path(self) -> Path:
        """获取Stanza模型路径"""
        home = Path.home()
        return home / 'stanza_resources' / 'en'
        
    def _check_model_exists(self) -> bool:
        """检查模型是否已下载"""
        model_path = self._get_stanza_model_path()
        return model_path.exists() and any(model_path.glob('*.pt'))
        
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
            logger.info("使用离线模式初始化Stanza")
            return stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse', download_method=None)
            
        for attempt in range(max_retries):
            try:
                # 检查模型是否已下载
                if not self._check_model_exists():
                    logger.info("正在下载Stanza模型...")
                    stanza.download('en')
                
                return stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse', download_method=DownloadMethod.REUSE_RESOURCES)
                
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
                    # 如果是名词、形容词或限定词，添加到当前短语
                    if word.pos in ['NOUN', 'PROPN', 'ADJ', 'DET']:
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
                    # 如果是动词、助动词或副词，添加到当前短语
                    if word.pos in ['VERB', 'AUX', 'ADV']:
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
            
            # 过滤掉太短或太长的短语，以及不合理的短语
            valid_phrases = []
            for phrase in phrases:
                words = phrase.split()
                # 短语长度在2-5个词之间
                if 2 <= len(words) <= 5:
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
            r'\b\w+\s+\w+\s+\w+\s+\w+\b',         # 较长的短语
            r'\b\w+\s+\w+\b',                      # 太短的短语
            r'\b[a-z]\s+[a-z]\b',                  # 单个字母的组合
            r'\b\d+\s+\w+\b',                      # 数字和词的组合
            r'\b\w+\s+\d+\b'                       # 词和数字的组合
        ]
        
        return not any(re.search(pattern, phrase.lower()) for pattern in invalid_patterns)
        
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
                if not text[0].isupper() and not text.startswith('"'):
                    current_sentence += " " + text
                    current_end = sub.end
                else:
                    # 保存当前句子并开始新的句子
                    if len(current_sentence.split()) <= 30:  # 限制句子长度
                        merged_sentences.append({
                            'text': current_sentence,
                            'start': current_start,
                            'end': current_end
                        })
                    current_sentence = text
                    current_start = sub.start
                    current_end = sub.end
        
        # 添加最后一个句子
        if current_sentence and len(current_sentence.split()) <= 30:
            merged_sentences.append({
                'text': current_sentence,
                'start': current_start,
                'end': current_end
            })
            
        return merged_sentences
        
    def get_cerf_level(self, word: str) -> str:
        lvl = cefr_analyzer.get_average_word_level_CEFR(word)
        if lvl:
            return str(lvl)
        else:
            return None
        
    def _is_difficult_word(self, word: str) -> bool:
        """
        使用CEFR判断单词是否高于B1难度
        
        Args:
            word: 要判断的单词
            
        Returns:
            是否高于B1难度
        """
        word = word.lower()
        try:
            level = self.get_cerf_level(word)
            # B1对应的CEFR等级是B1，高于B1的等级是B2, C1, C2
            return level in ['B2', 'C1', 'C2']
        except:
            # 如果单词不在词库中，使用备用规则
            return (
                len(word) > 8 or
                not word.isalpha() or
                word.endswith(('tion', 'sion', 'ment', 'ance', 'ence', 'ity', 'ness')) or
                word.startswith(('un', 'dis', 'in', 'im', 'ir', 'il'))
            )
        
    def _is_difficult_phrase(self, phrase: str) -> bool:
        """
        判断短语是否高于B1难度
        
        Args:
            phrase: 要判断的短语
            
        Returns:
            是否高于B1难度
        """
        # 检查短语中的单词难度
        words = phrase.split()
        difficult_words = sum(1 for word in words if self._is_difficult_word(word))
        
        # 如果短语包含多个难词，认为是难短语
        if difficult_words >= 2:
            return True
            
        # 检查短语长度
        if len(words) > 4:
            return True
            
        # 检查是否包含复杂结构
        complex_structures = [
            'as well as',
            'in order to',
            'so that',
            'such as',
            'due to',
            'in spite of',
            'as a result',
            'in addition to',
            'on the other hand',
            'in contrast to',
            'in terms of',
            'with respect to',
            'in accordance with',
            'in the event of',
            'for the purpose of'
        ]
        
        return any(struct in phrase.lower() for struct in complex_structures)
        
    def _is_difficult_sentence(self, sentence: str) -> bool:
        """
        判断句子是否高于B1难度
        
        Args:
            sentence: 要判断的句子
            
        Returns:
            是否高于B1难度
        """
        # 检查句子长度
        words = sentence.split()
        if len(words) > 20:
            return True
            
        # 检查难词数量
        difficult_words = sum(1 for word in words if self._is_difficult_word(word))
        if difficult_words > 3:
            return True
            
        # 检查是否包含复杂结构
        complex_structures = [
            'although',
            'despite',
            'however',
            'nevertheless',
            'therefore',
            'consequently',
            'furthermore',
            'moreover',
            'nonetheless',
            'whereas'
        ]
        
        return any(struct in sentence.lower() for struct in complex_structures)
        
    def analyze_difficulty(self):
        """
        分析所有句子的难度，并将结果写入文件
        """
        merged_sentences = self._merge_sentences()
        
        for sentence in merged_sentences:
            text = sentence['text']
            
            # 分析单词难度
            words = text.split()
            for word in words:
                if self._is_difficult_word(word):
                    self.difficult_words.add(word.lower())
            
            # 提取并分析短语难度
            phrases = self._extract_phrases(text)
            for phrase in phrases:
                if self._is_difficult_phrase(phrase):
                    self.difficult_phrases.add(phrase)
                # self.difficult_phrases.add(phrase)
            
            # 分析句子难度
            if self._is_difficult_sentence(text):
                self.difficult_sentences.add(text)
        
        # 将结果写入文件
        self._write_results()
        
    def _write_results(self):
        """将分析结果写入文件"""
        # 写入难词及其CEFR等级
        with open('generated/words.txt', 'w', encoding='utf-8') as f:
            for word in sorted(self.difficult_words):
                try:
                    level = self.get_cerf_level(word)
                    f.write(f"{word} (CEFR: {level})\n")
                except:
                    f.write(f"{word} (Unknown level)\n")
                
        # 写入难短语
        with open('generated/phrases.txt', 'w', encoding='utf-8') as f:
            for phrase in sorted(self.difficult_phrases):
                f.write(f"{phrase}\n")
                
        # 写入难句子
        with open('generated/sentences.txt', 'w', encoding='utf-8') as f:
            for sentence in sorted(self.difficult_sentences):
                f.write(f"{sentence}\n")
                
        logger.info("分析结果已写入文件")

def main():
    try:
        # 示例用法
        # srt_file = "demo.srt"
        srt_file = "JoeRogan-2294-GLT1061251245-2294 - Dr. Suzanne Humphries.srt"
        
        # 检查是否使用离线模式
        offline_mode = os.environ.get('STANZA_OFFLINE', 'false').lower() == 'true'
        analyzer = LanguageAnalyzer(srt_file, offline_mode=offline_mode)
        
        # 分析难度
        analyzer.analyze_difficulty()
        
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