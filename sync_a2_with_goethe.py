#!/usr/bin/env python3
"""
Sync A2.csv with official Goethe Institute A2 word list.
- Removes words not in the official list
- Adds missing words from the official list
- Uses English translations as temporary placeholders for missing Spanish translations
"""
import csv
import sys
import os
import argparse

# Constants
PLACEHOLDER_PREFIX = '[TODO'
DATA_DIR = 'data'  # Directory for intermediate data files

def ensure_data_dir():
    """Ensure data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def main(goethe_words_file=None, english_trans_file=None):
    """
    Sync A2.csv with official Goethe A2 word list.
    
    Args:
        goethe_words_file: Path to file with Goethe A2 words (one per line)
        english_trans_file: Path to file with English translations (word<tab>translation)
    """
    # Set default paths if not provided
    ensure_data_dir()
    if goethe_words_file is None:
        goethe_words_file = os.path.join(DATA_DIR, 'goethe_a2_words.txt')
    if english_trans_file is None:
        english_trans_file = os.path.join(DATA_DIR, 'a2_english_translations.txt')
    
    # Read current A2 with translations
    print("Reading current A2.csv...")
    with open('A2.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        current_a2 = {row['German_Lemma'].strip().lower(): row for row in reader}
    
    print(f"Current A2.csv has {len(current_a2)} words")
    
    # Read official Goethe A2 word list
    print("Reading official Goethe A2 word list...")
    if not os.path.exists(goethe_words_file):
        print(f"ERROR: Goethe word list file not found: {goethe_words_file}")
        print("Please provide the file using --goethe-words argument")
        return 1
        
    with open(goethe_words_file, 'r', encoding='utf-8') as f:
        goethe_words = [line.strip() for line in f if line.strip()]
    
    print(f"Official Goethe A2 list has {len(goethe_words)} words")
    
    # Read English translations from Goethe files (optional)
    print("Reading English translations...")
    english_trans = {}
    if os.path.exists(english_trans_file):
        with open(english_trans_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    word, trans = line.strip().split('\t', 1)
                    english_trans[word.lower()] = trans
        print(f"Loaded {len(english_trans)} English translations")
    else:
        print(f"English translations file not found (optional): {english_trans_file}")
    
    # Build new A2 list
    new_a2 = []
    missing_count = 0
    kept_count = 0
    
    for goethe_word in sorted(goethe_words, key=str.lower):
        word_lower = goethe_word.lower()
        
        if word_lower in current_a2:
            # Keep existing word with its translation if it's not a placeholder
            row = current_a2[word_lower]
            trans = row['Spanish_Translation']
            # Safely handle None values
            trans = trans.strip() if trans else ''
            
            # Skip placeholder translations
            if trans and not trans.startswith(PLACEHOLDER_PREFIX):
                # Use the Goethe word form (preserves correct capitalization)
                new_a2.append({
                    'German_Lemma': goethe_word,
                    'Spanish_Translation': trans
                })
                kept_count += 1
            else:
                # Add with empty translation (needs Spanish translation)
                new_a2.append({
                    'German_Lemma': goethe_word,
                    'Spanish_Translation': ''
                })
                missing_count += 1
        else:
            # Add missing word with empty translation (needs Spanish translation)
            new_a2.append({
                'German_Lemma': goethe_word,
                'Spanish_Translation': ''
            })
            missing_count += 1
    
    # Count removed words
    removed = set(current_a2.keys()) - {w.lower() for w in goethe_words}
    removed_count = len(removed)
    
    print(f"\nSummary:")
    print(f"  Kept (with Spanish translations): {kept_count}")
    print(f"  Added (need Spanish translation): {missing_count}")
    print(f"  Removed (not in Goethe): {removed_count}")
    print(f"  Total in new A2.csv: {len(new_a2)}")
    
    # Write new A2.csv
    print("\nWriting new A2.csv...")
    with open('A2.csv', 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['German_Lemma', 'Spanish_Translation']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_a2)
    
    print(f"Successfully updated A2.csv with {len(new_a2)} words from official Goethe list!")
    
    # Save list of words that need Spanish translation
    if missing_count > 0:
        output_file = os.path.join(DATA_DIR, 'a2_words_need_spanish_translation.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            for row in new_a2:
                if not row['Spanish_Translation']:
                    # Add English translation as reference if available
                    eng_trans = english_trans.get(row['German_Lemma'].lower(), '')
                    f.write(f"{row['German_Lemma']}\t{eng_trans}\n")
        print(f"\n⚠ NOTE: {missing_count} words have empty Spanish translations")
        print(f"         These need to be translated from English/German to Spanish")
        print(f"         List saved to {output_file}")
    
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sync A2.csv with official Goethe A2 word list')
    parser.add_argument('--goethe-words', help='Path to Goethe A2 words file')
    parser.add_argument('--english-trans', help='Path to English translations file')
    args = parser.parse_args()
    
    sys.exit(main(args.goethe_words, args.english_trans))
