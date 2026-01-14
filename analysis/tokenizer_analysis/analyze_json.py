import json
import argparse
from collections import defaultdict

# Define the Unicode ranges for various scripts and their subcategories.
SCRIPT_RANGES = {
    "Chinese": {
        "Unified Ideographs": [(0x4E00, 0x9FFF)],
        "Extension A": [(0x3400, 0x4DBF)],
        "Extension B": [(0x20000, 0x2A6DF)],
        "Extension C": [(0x2A700, 0x2B73F)],
        "Extension D": [(0x2B740, 0x2B81F)],
        "Extension E": [(0x2B820, 0x2CEAF)],
        "Extension F": [(0x2CEB0, 0x2EBEF)],
        "Extension G": [(0x30000, 0x3134F)],
        "Extension H": [(0x31350, 0x323AF)],
    },
    "Korean": {
        "Hangul Syllables": [(0xAC00, 0xD7A3)],
        "Hangul Jamo": [(0x1100, 0x11FF)],
        "Hangul Jamo Extended-A": [(0xA960, 0xA97F)],
        "Hangul Jamo Extended-B": [(0xD7B0, 0xD7FF)],
    },
    "Japanese": {
        "Hiragana": [(0x3040, 0x309F)],
        "Katakana": [(0x30A0, 0x30FF)],
    },
    "Arabic (incl. Urdu)": {
        "Arabic": [(0x0600, 0x06FF)],
        "Arabic Supplement": [(0x0750, 0x077F)],
        "Arabic Extended-A": [(0x08A0, 0x08FF)],
    },
    "Cyrillic": {
        "Cyrillic": [(0x0400, 0x04FF)],
        "Cyrillic Supplement": [(0x0500, 0x052F)],
    },
    "Bengali": {
        "Bengali": [(0x0980, 0x09FF)],
    },
    "Shan": {
        "Myanmar (incl. Shan chars)": [(0x1000, 0x109F)],
        "Shan Vowels/Tones (Ext-A)": [(0xA9E0, 0xA9FF)],
        "Shan Tones (Ext-B)": [(0xA980, 0xA9DF)],
    },
    "Hindi (Devanagari)": {
        "Devanagari": [(0x0900, 0x097F)],
    },
    "Greek": {
        "Greek and Coptic": [(0x0370, 0x03FF)],
    },
    "Thai": {
        "Thai": [(0x0E00, 0x0E7F)],
    },
    "Hebrew": {
        "Hebrew": [(0x0590, 0x05FF)],
    }
}

def analyze_script(script_name, script_categories, categories_data):
    """
    Analyzes a single script's presence in the vocabulary.
    """
    print(f"\n--- {script_name} Character Analysis ---")

    all_possible_chars = set()
    for subcat_ranges in script_categories.values():
        for start, end in subcat_ranges:
            for char_code in range(start, end + 1):
                try:
                    all_possible_chars.add(chr(char_code))
                except ValueError:
                    continue
    
    initial_possible_chars = len(all_possible_chars)

    encountered_chars = set()
    category_key = script_name.split(' ')[0].lower()
    
    # **MODIFIED LOGIC:** Check specific category, then 'misc', then 'vocab'
    token_list = []
    if category_key in categories_data:
        token_list = categories_data[category_key]
    elif "misc" in categories_data:
        print(f"(Note: No specific '{category_key}' category found, analyzing 'misc' category instead)")
        token_list = categories_data["misc"]
    else:
        print(f"(Note: No specific '{category_key}' or 'misc' category found, analyzing entire vocab)")
        token_list = categories_data.get("vocab", [])

    flat_ranges = [r for sub_ranges in script_categories.values() for r in sub_ranges]

    for token_data in token_list:
        for char in token_data["token"]:
            codepoint = ord(char)
            if any(start <= codepoint <= end for start, end in flat_ranges):
                encountered_chars.add(char)

    remaining_chars = all_possible_chars - encountered_chars
    
    encountered_count = len(encountered_chars)
    remaining_count = len(remaining_chars)
    
    percentage_encountered = (encountered_count / initial_possible_chars) * 100 if initial_possible_chars > 0 else 0
    percentage_remaining = (remaining_count / initial_possible_chars) * 100 if initial_possible_chars > 0 else 0

    print(f"- Total possible unique characters in script: {initial_possible_chars}")
    print(f"- Unique characters encountered in tokens: {encountered_count} ({percentage_encountered:.4f}%)")
    print(f"- Unique characters NOT encountered: {remaining_count} ({percentage_remaining:.4f}%)")
    
    filename_base = script_name.split(' ')[0].lower()
    with open(f"{filename_base}_encountered.txt", "w", encoding="utf-8") as f:
        for char in sorted(list(encountered_chars)):
            f.write(char + "\n")
            
    with open(f"{filename_base}_not_encountered.txt", "w", encoding="utf-8") as f:
        for char in sorted(list(remaining_chars)):
            f.write(char + "\n")

def analyze_categories(json_file):
    """
    Analyzes categorized tokens from a JSON file for various scripts.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        categories = json.load(f)

    total_tokens = sum(len(tokens) for tokens in categories.values())

    print("--- Token Category Statistics ---")
    for category, tokens in categories.items():
        count = len(tokens)
        percentage = (count / total_tokens) * 100 if total_tokens > 0 else 0
        print(f"- {category.capitalize()}: {count} tokens ({percentage:.2f}%)")

    for script_name, script_categories in SCRIPT_RANGES.items():
        analyze_script(script_name, script_categories, categories)

    print("\nCharacter lists for all analyzed scripts have been written to their respective files.")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze categorized token JSON file for character coverage across multiple scripts."
    )
    parser.add_argument("json_file", help="Path to the JSON file to analyze.")
    args = parser.parse_args()

    analyze_categories(args.json_file)

if __name__ == "__main__":
    main()
