#!/usr/bin/env python3
"""
Fix B2.csv to contain only B2 level German words from master_lemmas.txt.

B2 words are defined as:
- Words in master_lemmas.txt
- NOT in A1.csv
- NOT in A2.csv
- NOT in B1.csv
- NOT in C1.csv
"""

import pandas as pd
import sys

def fix_b2():
    print("Loading data files...")
    
    # Load all level CSVs
    a1_df = pd.read_csv('A1.csv', encoding='utf-8')
    a2_df = pd.read_csv('A2.csv', encoding='utf-8')
    b1_df = pd.read_csv('B1.csv', encoding='utf-8')
    b2_df = pd.read_csv('B2.csv', encoding='utf-8')
    c1_df = pd.read_csv('C1.csv', encoding='utf-8')
    
    # Extract word sets
    a1 = set(a1_df['German_Lemma'])
    a2 = set(a2_df['German_Lemma'])
    b1 = set(b1_df['German_Lemma'])
    b2_current = set(b2_df['German_Lemma'])
    c1 = set(c1_df['German_Lemma'])
    
    # Load master lemmas (only non-empty lines after stripping whitespace)
    with open('master_lemmas.txt', 'r', encoding='utf-8') as f:
        master = set(line.strip() for line in f if line.strip())
    
    print(f"\nCurrent state:")
    print(f"  Master lemmas: {len(master)}")
    print(f"  A1: {len(a1)}")
    print(f"  A2: {len(a2)}")
    print(f"  B1: {len(b1)}")
    print(f"  B2: {len(b2_current)}")
    print(f"  C1: {len(c1)}")
    
    # Calculate what B2 should contain
    should_be_b2 = master - a1 - a2 - b1 - c1
    
    print(f"\nTarget B2 word count: {len(should_be_b2)}")
    
    # Find what needs to be removed
    to_remove = b2_current - should_be_b2
    
    print(f"\nWords to remove from B2: {len(to_remove)}")
    
    # Analyze removals
    not_in_master = to_remove - master
    duplicate_with_c1 = to_remove & c1
    
    print(f"  - Not in master_lemmas.txt: {len(not_in_master)}")
    print(f"  - Duplicate with C1: {len(duplicate_with_c1)}")
    
    if to_remove:
        print(f"\nFirst 20 words to remove:")
        for i, word in enumerate(sorted(to_remove)[:20], 1):
            reason = []
            if word not in master:
                reason.append("not in master")
            if word in c1:
                reason.append("in C1")
            print(f"  {i}. {word} ({', '.join(reason)})")
    
    # Create new B2 dataframe with only correct words
    # Keep original translations from B2 where available
    new_b2_df = b2_df[b2_df['German_Lemma'].isin(should_be_b2)].copy()
    
    # Sort alphabetically
    new_b2_df = new_b2_df.sort_values('German_Lemma').reset_index(drop=True)
    
    print(f"\nNew B2.csv will have {len(new_b2_df)} words")
    
    # Save the fixed B2.csv
    new_b2_df.to_csv('B2.csv', index=False, encoding='utf-8')
    
    print(f"\n✓ B2.csv has been updated!")
    print(f"  Removed: {len(to_remove)} words")
    print(f"  Kept: {len(new_b2_df)} words")
    
    # Verify
    final_b2 = set(new_b2_df['German_Lemma'])
    if final_b2 == should_be_b2:
        print("\n✓ Verification successful: B2.csv now contains exactly the correct B2 words!")
        return 0
    else:
        print("\n✗ Verification failed!")
        missing = should_be_b2 - final_b2
        extra = final_b2 - should_be_b2
        if missing:
            print(f"  Missing {len(missing)} words that should be in B2")
        if extra:
            print(f"  Has {len(extra)} extra words")
        return 1

if __name__ == '__main__':
    sys.exit(fix_b2())
