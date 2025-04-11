# Language Analyzer

This program analyzes SRT subtitle files to identify words, phrases, and sentences that might be difficult for B1 level language learners. It provides phonetic transcriptions and Chinese translations for difficult words, and counts their frequency of occurrence.

## Features

- Identifies difficult words, phrases, and sentences for B1 level learners
- Provides phonetic transcriptions for difficult words
- Provides Chinese translations
- Counts frequency of difficult elements
- Handles SRT subtitle files

## Requirements

- Python 3.6 or higher
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

Download model for English

```bash
python -c "import stanza; stanza.download('en')"
```

Fortunately, huggingface site provides a vital path to download the model https://huggingface.co/stanfordnlp/stanza-en

## Usage

1. Run the program:
```bash
python language_analyzer.py
```

2. When prompted, enter the path to your SRT file

3. The program will analyze the file and display:
   - Difficult words with phonetic transcriptions and Chinese translations
   - Difficult phrases with Chinese translations
   - Difficult sentences with Chinese translations
   - Frequency counts for each element

## Notes

- The program uses a simplified algorithm to determine difficulty level
- For more accurate results, you may want to customize the difficulty criteria in the code
- Internet connection is required for translations 