import unittest
import os
import tempfile
import shutil
from pathlib import Path
import asyncio

from language_analyzer import LanguageAnalyzer, clean_text, ensure_dir_exists

class TestLanguageAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """测试前的准备工作"""
        # 创建临时目录
        cls.temp_dir = tempfile.mkdtemp()
        # 创建测试用的demo.srt文件
        cls.demo_srt = os.path.join(cls.temp_dir, "demo.srt")
        with open(cls.demo_srt, "w", encoding="utf-8") as f:
            f.write("""1
00:00:00,000 --> 00:00:05,000
Hello world.

2
00:00:05,000 --> 00:00:10,000
This is a test.
""")
        
    @classmethod
    def tearDownClass(cls):
        """测试后的清理工作"""
        # 删除临时目录
        shutil.rmtree(cls.temp_dir)
        
    def test_clean_text(self):
        """测试文本清理功能"""
        # 测试清理HTML标签
        text = "<p>Hello <b>world</b></p>"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "Hello world")
        
        # 测试清理特殊字符
        text = "Hello, world! @#$%"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "Hello world")
        
    def test_ensure_dir_exists(self):
        """测试目录创建功能"""
        test_dir = Path("test-dir")
        try:
            # 确保目录不存在
            if test_dir.exists():
                test_dir.rmdir()
                
            # 测试创建目录
            created_dir = ensure_dir_exists(test_dir)
            self.assertTrue(created_dir.exists())
            self.assertTrue(created_dir.is_dir())
            
        finally:
            # 清理
            if test_dir.exists():
                test_dir.rmdir()
                
    def test_analyze_subtitle(self):
        """测试字幕分析功能"""
        analyzer = LanguageAnalyzer(self.demo_srt)
        # 创建事件循环并运行异步测试
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(analyzer.analyze_subtitle(analyzer.subtitles[0]))
        print(f'result: {result}')
        
        # 检查基本字段
        self.assertIn('original_text', result)
        self.assertIn('cleaned_text', result)
        self.assertIn('word_count', result)
        self.assertIn('sentence_count', result)
        
        # 检查清理后的文本
        self.assertEqual(result['cleaned_text'], "Hello world")

if __name__ == '__main__':
    unittest.main() 