import pandas as pd
import os

LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1']
FILES = {level: f"{level}.csv" for level in LEVELS}
MASTER_FILE = "master_lemmas.txt"

# Known typos map (bad -> good)
TYPO_FIXES = {
    'ait': 'alt',
    'aiter': 'Alter',
    'aberjetzt': 'aber jetzt',
    'doiiar': 'Dollar',
    'endiiich': 'endlich',
    'endiich': 'endlich',
    'fruehling': 'Frühling',
    'fuer': 'für',
    'fuesse': 'Füße',
    'fuessen': 'Füßen',
    'gefaehrdet': 'gefährdet',
    'gefaengnis': 'Gefängnis',
    'gefaellt': 'gefällt',
    'gehoert': 'gehört',
    'gruen': 'grün',
    'gruesse': 'Grüße',
    'hallo': 'Hallo',
    'hoeren': 'hören',
    'koennen': 'können',
    'koenig': 'König',
    'koerper': 'Körper',
    'kueche': 'Küche',
    'kuehl': 'kühl',
    'kuenstler': 'Künstler',
    'maedchen': 'Mädchen',
    'maenner': 'Männer',
    'moechte': 'möchte',
    'moegen': 'mögen',
    'muessen': 'müssen',
    'muetter': 'Mütter',
    'natuerlich': 'natürlich',
    'oeffnen': 'öffnen',
    'oel': 'Öl',
    'oesterreich': 'Österreich',
    'ploezlich': 'plötzlich',
    'spaet': 'spät',
    'spaeter': 'später',
    'schoen': 'schön',
    'schueler': 'Schüler',
    'tuer': 'Tür',
    'ueber': 'über',
    'ueberall': 'überall',
    'uebung': 'Übung',
    'waehrend': 'während',
    'waere': 'wäre',
    'wuerde': 'würde',
    'wuenschen': 'wünschen',
    'zaehne': 'Zähne',
    'zurueck': 'zurück',
    'zwoelf': 'zwölf'
}

def load_master_lemmas():
    if not os.path.exists(MASTER_FILE):
        print("Master lemmas file not found!")
        return set()
    with open(MASTER_FILE, 'r', encoding='utf-8') as f:
        # Create a set of valid lemmas
        return set(line.strip() for line in f if line.strip())

def restore_umlauts(word, master_set):
    # Try common replacements
    replacements = [
        ('ae', 'ä'),
        ('ue', 'ü'),
        ('oe', 'ö'),
        ('ss', 'ß')
    ]
    
    # Generate candidates
    candidates = [word]
    for old, new in replacements:
        next_candidates = []
        for cand in candidates:
            next_candidates.append(cand)
            if old in cand:
                next_candidates.append(cand.replace(old, new))
        candidates = next_candidates
    
    # Check if any candidate exists in master set (case-insensitive check might be needed?)
    # For now, check exact match or capitalized match in master set
    for cand in candidates:
        # Check exact
        if cand in master_set:
            return cand
        # Check capitalized (only uppercase first letter, preserve rest)
        capitalized = cand[0].upper() + cand[1:] if cand else cand
        if capitalized in master_set:
            return capitalized
            
    return word

def fix_csvs():
    print("Loading master lemmas...")
    master_lemmas = load_master_lemmas()
    print(f"Loaded {len(master_lemmas)} master lemmas.")
    
    for level in LEVELS:
        filename = FILES[level]
        if not os.path.exists(filename):
            continue
            
        print(f"\nProcessing {filename}...")
        try:
            df = pd.read_csv(filename, encoding='utf-8')
            fixed_count = 0
            
            new_rows = []
            
            for idx, row in df.iterrows():
                lemma = str(row['German_Lemma']).strip()
                trans = str(row['Spanish_Translation']).strip()
                
                original_lemma = lemma
                
                # 1. Fix Typos
                if lemma.lower() in TYPO_FIXES:
                    lemma = TYPO_FIXES[lemma.lower()]
                
                # 2. Restore Umlauts
                # Only try if not in master set already
                if lemma not in master_lemmas:
                    lemma = restore_umlauts(lemma, master_lemmas)
                
                # 3. Fix Capitalization
                # If lemma is lowercase but exists capitalized in master, capitalize it
                # Only uppercase first letter, preserve rest of word
                if lemma[0].islower():
                    capitalized = lemma[0].upper() + lemma[1:]
                    if capitalized in master_lemmas:
                        lemma = capitalized
                
                # Track changes
                if lemma != original_lemma:
                    fixed_count += 1
                    # print(f"  Fixed: {original_lemma} -> {lemma}")
                
                new_rows.append({'German_Lemma': lemma, 'Spanish_Translation': trans})
            
            # Save fixed file
            fixed_df = pd.DataFrame(new_rows)
            fixed_filename = f"{level}_fixed.csv"
            fixed_df.to_csv(fixed_filename, index=False, encoding='utf-8')
            print(f"Saved {fixed_filename}. Fixed {fixed_count} entries.")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    fix_csvs()
