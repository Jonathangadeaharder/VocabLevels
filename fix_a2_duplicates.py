#!/usr/bin/env python3
"""
Script to remove A2 words from C1.csv to ensure each word appears in only one level.
"""
import csv
import sys

def main():
    # Read A2 words
    print("Reading A2.csv...")
    with open('A2.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        a2_words = {row['German_Lemma'].strip() for row in reader}
    
    print(f"Found {len(a2_words)} words in A2.csv")
    
    # Read C1 words
    print("\nReading C1.csv...")
    with open('C1.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        c1_data = [row for row in reader]
    
    print(f"Found {len(c1_data)} words in C1.csv")
    
    # Find duplicates
    duplicates = []
    for row in c1_data:
        word = row['German_Lemma'].strip()
        if word in a2_words:
            duplicates.append(word)
    
    print(f"\nFound {len(duplicates)} A2 words in C1.csv:")
    for word in sorted(duplicates):
        print(f"  - {word}")
    
    if not duplicates:
        print("\nNo duplicates found. C1.csv is clean!")
        return 0
    
    # Remove duplicates from C1
    print(f"\nRemoving {len(duplicates)} duplicates from C1.csv...")
    c1_cleaned = [row for row in c1_data if row['German_Lemma'].strip() not in a2_words]
    
    print(f"C1.csv will have {len(c1_cleaned)} words after cleanup (was {len(c1_data)})")
    
    # Write cleaned C1
    with open('C1.csv', 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['German_Lemma', 'Spanish_Translation']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(c1_cleaned)
    
    print(f"\nSuccessfully cleaned C1.csv! Removed {len(duplicates)} A2 words.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
