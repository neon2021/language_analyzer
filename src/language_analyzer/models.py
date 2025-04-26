from dataclasses import dataclass
from typing import List, Optional
from datetime import timedelta

@dataclass
class SubtitleEntry:
    """
    字幕条目数据类
    """
    index: int
    start_time: timedelta
    end_time: timedelta
    text: str
    cleaned_text: str
    word_count: int
    sentence_count: int
    avg_word_length: float
    complexity_score: float
    language: str
    confidence: float

@dataclass
class AnalysisResult:
    """
    分析结果数据类
    """
    total_entries: int
    total_words: int
    total_sentences: int
    avg_word_length: float
    avg_sentence_length: float
    avg_complexity: float
    language_distribution: dict
    entries: List[SubtitleEntry]
    error_count: int
    success_rate: float 