#!/usr/bin/env python3
"""
Fill in missing Spanish translations in A2.csv by looking them up in other level files.
"""
import csv
import sys
import os

# Constants
DATA_DIR = 'data'
LEVEL_FILES = ['A1', 'B1', 'B2', 'C1']

def ensure_data_dir():
    """Ensure data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def main():
    # Load translations from other levels
    print("Loading Spanish translations from other level files...")
    translations = {}
    for level in LEVEL_FILES:
        level_file = f'{level}.csv'
        if not os.path.exists(level_file):
            print(f"Warning: {level_file} not found, skipping...")
            continue
            
        with open(level_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row['German_Lemma'].strip()
                word_lower = word.lower()
                trans = row['Spanish_Translation'].strip()
                if trans and word_lower not in translations:  # Keep first occurrence
                    # Store both the translation and original capitalization
                    translations[word_lower] = (trans, word)
    
    print(f"Found {len(translations)} translations in other level files")
    
    # Update A2.csv
    print("Updating A2.csv with found translations...")
    with open('A2.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    updated_count = 0
    still_empty_count = 0
    
    for row in rows:
        word = row['German_Lemma'].strip()
        word_lower = word.lower()
        if not row['Spanish_Translation']:
            # Look up translation (case-insensitive)
            if word_lower in translations:
                trans, _ = translations[word_lower]
                row['Spanish_Translation'] = trans
                updated_count += 1
            else:
                still_empty_count += 1
    
    # Write updated A2.csv
    with open('A2.csv', 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['German_Lemma', 'Spanish_Translation']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nResults:")
    print(f"  Filled in {updated_count} Spanish translations from other levels")
    print(f"  {still_empty_count} words still need Spanish translation")
    
    # Save list of words still needing translation
    if still_empty_count > 0:
        ensure_data_dir()
        output_file = os.path.join(DATA_DIR, 'a2_still_need_translation.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            for row in rows:
                if not row['Spanish_Translation']:
                    f.write(f"{row['German_Lemma']}\n")
        print(f"  List saved to {output_file}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
