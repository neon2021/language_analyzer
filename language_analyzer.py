import pysrt
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import re
from googletrans import Translator
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class LanguageAnalyzer:
    def __init__(self):
        self.translator = Translator()
        self.b1_level_words = set()  # This would be populated with B1 level vocabulary
        self.difficult_words = Counter()
        self.difficult_phrases = Counter()
        self.difficult_sentences = Counter()
        
    def load_srt(self, file_path):
        """Load and parse SRT file"""
        return pysrt.open(file_path)
    
    def is_difficult_word(self, word):
        """Check if a word is difficult for B1 level"""
        # This is a simplified version. In a real implementation, you would:
        # 1. Check against a B1 vocabulary list
        # 2. Consider word frequency
        # 3. Check word length and complexity
        word = word.lower()
        return (len(word) > 8 or 
                not word.isalpha() or 
                word not in self.b1_level_words)
    
    def is_difficult_phrase(self, phrase):
        """Check if a phrase is difficult for B1 level"""
        # This would check for:
        # 1. Idiomatic expressions
        # 2. Complex grammatical structures
        # 3. Cultural references
        return len(phrase.split()) > 3
    
    def is_difficult_sentence(self, sentence):
        """Check if a sentence is difficult for B1 level"""
        # This would check for:
        # 1. Sentence length
        # 2. Complex grammatical structures
        # 3. Number of difficult words
        words = word_tokenize(sentence)
        difficult_words = sum(1 for word in words if self.is_difficult_word(word))
        return difficult_words > 3 or len(words) > 20
    
    def get_phonetic(self, word):
        """Get phonetic transcription of a word"""
        # This is a placeholder. In a real implementation, you would:
        # 1. Use a dictionary API
        # 2. Use a phonetic transcription library
        return f"/{word}/"
    
    def get_translation(self, text):
        """Get Chinese translation of text"""
        try:
            translation = self.translator.translate(text, dest='zh-cn')
            return translation.text
        except:
            return "Translation not available"
    
    def analyze_srt(self, file_path):
        """Analyze SRT file and identify difficult elements"""
        subs = self.load_srt(file_path)
        
        for sub in subs:
            text = sub.text
            # Remove HTML tags if present
            text = re.sub(r'<[^>]+>', '', text)
            
            # Analyze words
            words = word_tokenize(text)
            for word in words:
                if self.is_difficult_word(word):
                    self.difficult_words[word.lower()] += 1
            
            # Analyze phrases
            phrases = text.split('.')
            for phrase in phrases:
                if self.is_difficult_phrase(phrase):
                    self.difficult_phrases[phrase.strip()] += 1
            
            # Analyze sentences
            sentences = sent_tokenize(text)
            for sentence in sentences:
                if self.is_difficult_sentence(sentence):
                    self.difficult_sentences[sentence.strip()] += 1
    
    def print_results(self):
        """Print analysis results"""
        print("\nDifficult Words:")
        print("-" * 50)
        for word, count in self.difficult_words.most_common():
            phonetic = self.get_phonetic(word)
            translation = self.get_translation(word)
            print(f"{word} ({phonetic}) - {translation} - Frequency: {count}")
        
        print("\nDifficult Phrases:")
        print("-" * 50)
        for phrase, count in self.difficult_phrases.most_common():
            translation = self.get_translation(phrase)
            print(f"{phrase} - {translation} - Frequency: {count}")
        
        print("\nDifficult Sentences:")
        print("-" * 50)
        for sentence, count in self.difficult_sentences.most_common():
            translation = self.get_translation(sentence)
            print(f"{sentence} - {translation} - Frequency: {count}")

def main():
    analyzer = LanguageAnalyzer()
    srt_file = input("Please enter the path to your SRT file: ")
    
    if not os.path.exists(srt_file):
        print("File not found!")
        return
    
    analyzer.analyze_srt(srt_file)
    analyzer.print_results()

if __name__ == "__main__":
    main() 