import pandas as pd
import re
import os

LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1']
FILES = {level: f"{level}_fixed.csv" for level in LEVELS}

# Common English words that often leak into subtitle scrapes
ENGLISH_STOPWORDS = {
    'the', 'and', 'of', 'to', 'in', 'is', 'you', 'that', 'it', 'he', 'was', 'for', 'on', 
    'are', 'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from', 'or', 
    'one', 'had', 'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when', 
    'your', 'can', 'said', 'there', 'use', 'an', 'each', 'which', 'she', 'do', 'how', 
    'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them', 
    'these', 'so', 'some', 'her', 'would', 'make', 'like', 'him', 'into', 'time', 
    'has', 'look', 'two', 'more', 'write', 'go', 'see', 'number', 'no', 'way', 
    'could', 'people', 'my', 'than', 'first', 'water', 'been', 'call', 'who', 'oil', 
    'its', 'now', 'find'
}

# Valid single letter or very short German words
VALID_SHORT = {'da', 'du', 'er', 'es', 'ja', 'ne', 'so', 'wo', 'zu', 'ab', 'an', 'im', 'in', 'ob', 'um', 'ei', 'po'}

def load_master_lemmas():
    if not os.path.exists("master_lemmas.txt"):
        return set()
    with open("master_lemmas.txt", 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def check_quality():
    print("=== MANUAL QUALITY CHECK REPORT ===")
    
    master_lemmas = load_master_lemmas()
    print(f"Loaded {len(master_lemmas)} master lemmas for whitelisting.")

    total_issues = 0
    
    for level in LEVELS:
        filename = FILES[level]
        if not os.path.exists(filename):
            continue
            
        print(f"\nChecking {filename}...")
        try:
            df = pd.read_csv(filename, encoding='utf-8')
            issues = []
            
            for idx, row in df.iterrows():
                lemma = str(row['German_Lemma']).strip()
                trans = str(row['Spanish_Translation']).strip()
                
                # 1. Check Empty
                if not lemma or lemma == 'nan':
                    issues.append((idx, lemma, "Empty Lemma"))
                    continue
                    
                # 2. Check Length & Garbage
                if len(lemma) < 2 and lemma.lower() not in VALID_SHORT:
                    issues.append((idx, lemma, "Too short (Garbage?)"))
                
                # 3. Check Characters (Numbers, Special)
                if re.search(r'[0-9]', lemma):
                    issues.append((idx, lemma, "Contains Numbers"))
                if re.search(r'[?!@#$%^&*()_=+\[\]{};:\'\"\\|<>~`\.]', lemma):
                    issues.append((idx, lemma, "Contains Special Chars"))
                    
                # 4. English Leakage
                if lemma.lower() in ENGLISH_STOPWORDS:
                    valid_homographs = {
                        'in', 'an', 'so', 'im', 'die', 'das', 'der', 'den', 'dem', 'des',
                        'art', 'bad', 'bar', 'boot', 'elf', 'hut', 'kind', 'rat', 'rot', 
                        'tag', 'war', 'wer', 'wo', 'nun', 'man', 'fast', 'rot', 'fest', 'rock',
                        'mist', 'gift', 'hand', 'wand', 'wind', 'arm', 'bank', 'bitten', 'bring',
                        'brutal', 'bunker', 'bus', 'butter', 'chef', 'chin', 'club', 'cool', 'deck',
                        'finger', 'fucking', 'gang', 'gas', 'general', 'gold', 'grab', 'grass', 'gut',
                        'halt', 'hammer', 'hier', 'hose', 'hund', 'hut', 'ideal', 'kinn', 'land',
                        'last', 'lied', 'links', 'list', 'machen', 'macht', 'mail', 'mama', 'mann',
                        'mark', 'mast', 'matt', 'me', 'meer', 'mehl', 'mein', 'mich', 'mild', 'moment',
                        'mond', 'mond', 'moor', 'mord', 'most', 'musk', 'muss', 'mutter', 'nach',
                        'name', 'nape', 'nein', 'nest', 'nett', 'neun', 'news', 'nie', 'not', 'nun',
                        'nur', 'nuss', 'nut', 'paar', 'pack', 'pan', 'park', 'part', 'pass', 'pause',
                        'plan', 'plus', 'po', 'post', 'pro', 'punk', 'pure', 'qualm', 'quiz', 'radio',
                        'rang', 'rank', 'rate', 'ratte', 'raum', 'raw', 'rede', 'rein', 'reis', 'reise',
                        'reit', 'rest', 'ring', 'rock', 'roh', 'roll', 'rom', 'rost', 'ruf', 'rund',
                        'sage', 'sake', 'salz', 'sand', 'sang', 'sank', 'satin', 'satt', 'satz', 'sau',
                        'schmal', 'schur', 'see', 'sehr', 'sein', 'send', 'set', 'shame', 'sink', 'sofa',
                        'sohn', 'sold', 'solo', 'span', 'spar', 'spat', 'speck', 'spin', 'spit', 'sport',
                        'spot', 'spur', 'sputter', 'stab', 'sack', 'sage', 'sake', 'sale', 'salt', 'same',
                        'sand', 'sane', 'sang', 'sank', 'sate', 'save', 'saw', 'scam', 'scan', 'scar',
                        'scat', 'school', 'scum', 'sea', 'seal', 'seam', 'seat', 'sect', 'see', 'seed',
                        'seek', 'seem', 'seen', 'seep', 'self', 'sell', 'send', 'sent', 'set', 'sew',
                        'shack', 'shad', 'ag', 'akt', 'angel', 'bald', 'ball', 'band', 'bang'
                    }
                    
                    if lemma.lower() not in valid_homographs and lemma not in master_lemmas:
                         issues.append((idx, lemma, "Possible English Word"))

                # 5. Empty Translation (Critical)
                if not trans or trans == 'nan':
                    issues.append((idx, lemma, "MISSING TRANSLATION"))
                
                # 6. Lowercase Lemma (Potential Noun Issue)
                if lemma[0].islower():
                    # Only flag if NOT in master lemmas
                    if lemma not in master_lemmas:
                        issues.append((idx, lemma, "Lowercase Lemma (Not in Master)"))

                # 7. Umlaut Replacements (ae, ue, oe)
                if 'ae' in lemma or 'ue' in lemma or 'oe' in lemma:
                    if lemma not in master_lemmas:
                        issues.append((idx, lemma, "Potential Umlaut Replacement (ae/ue/oe)"))

            # Print top issues for this file
            if issues:
                total_issues += len(issues)
                # Print first 50 issues
                for idx, lemma, reason in issues[:50]:
                    print(f"  Row {idx+2}: [{lemma}] -> {reason}")
                if len(issues) > 50:
                    print(f"  ... and {len(issues)-50} more.")
            else:
                print("  Clean.")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
                
    print(f"\nTotal Suspicious Entries Found: {total_issues}")

if __name__ == "__main__":
    check_quality()
