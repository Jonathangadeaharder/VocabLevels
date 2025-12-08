#!/usr/bin/env python3
"""
Sync A2.csv with official Goethe Institute A2 word list.
- Removes words not in the official list
- Adds missing words from the official list
- Uses English translations as temporary placeholders for missing Spanish translations
"""
import csv
import sys

def main():
    # Read current A2 with translations
    print("Reading current A2.csv...")
    with open('A2.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        current_a2 = {row['German_Lemma'].strip().lower(): row for row in reader}
    
    print(f"Current A2.csv has {len(current_a2)} words")
    
    # Read official Goethe A2 word list
    print("Reading official Goethe A2 word list...")
    with open('/tmp/goethe_a2_words.txt', 'r', encoding='utf-8') as f:
        goethe_words = [line.strip() for line in f if line.strip()]
    
    print(f"Official Goethe A2 list has {len(goethe_words)} words")
    
    # Read English translations from Goethe files
    print("Reading English translations...")
    english_trans = {}
    with open('/tmp/a2_english_translations.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                word, trans = line.strip().split('\t', 1)
                english_trans[word.lower()] = trans
    
    print(f"Loaded {len(english_trans)} English translations")
    
    # Build new A2 list
    new_a2 = []
    missing_count = 0
    kept_count = 0
    english_placeholder_count = 0
    
    for goethe_word in sorted(goethe_words, key=str.lower):
        word_lower = goethe_word.lower()
        
        if word_lower in current_a2:
            # Keep existing word with its translation if it's not a placeholder
            row = current_a2[word_lower]
            trans = row['Spanish_Translation']
            
            # Skip placeholder translations
            if trans and not trans.startswith('[TODO'):
                # Capitalize first letter for nouns (simple heuristic)
                if row['German_Lemma'][0].isupper():
                    lemma = row['German_Lemma']
                else:
                    lemma = goethe_word
                new_a2.append({
                    'German_Lemma': lemma,
                    'Spanish_Translation': trans
                })
                kept_count += 1
            else:
                # Treat as missing - add with empty translation
                lemma = goethe_word
                new_a2.append({
                    'German_Lemma': lemma,
                    'Spanish_Translation': ''
                })
                missing_count += 1
                english_placeholder_count += 1
        else:
            # Add missing word with empty translation (needs Spanish translation)
            lemma = goethe_word
            new_a2.append({
                'German_Lemma': lemma,
                'Spanish_Translation': ''  # Empty, needs Spanish translation
            })
            missing_count += 1
            english_placeholder_count += 1
    
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
    if english_placeholder_count > 0:
        with open('/tmp/a2_words_need_spanish_translation.txt', 'w', encoding='utf-8') as f:
            for row in new_a2:
                if not row['Spanish_Translation']:
                    # Add English translation as reference
                    eng_trans = english_trans.get(row['German_Lemma'].lower(), '')
                    f.write(f"{row['German_Lemma']}\t{eng_trans}\n")
        print(f"\n⚠ NOTE: {english_placeholder_count} words have empty Spanish translations")
        print(f"         These need to be translated from English/German to Spanish")
        print(f"         List saved to /tmp/a2_words_need_spanish_translation.txt")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
