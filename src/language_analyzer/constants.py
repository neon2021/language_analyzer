"""
常量定义
"""

FONT_PATHS = {
    'source_han_sans': [
        "/System/Library/Fonts/SourceHanSansTC-Regular.otf",  # macOS
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
        "C:/Windows/Fonts/SourceHanSansTC-Regular.otf"  # Windows
    ],
    'arial_unicode': [
        "/Library/Fonts/Arial Unicode.ttf"  # macOS
    ]
}

REQUIRED_FILES = [
    'default.zip',
    'tokenize/combined.pt',
    'pos/combined_charlm.pt',
    'lemma/combined_nocharlm.pt',
    'depparse/combined_charlm.pt'
] 