#!/usr/bin/env python3
"""
Fix capitalization in A2.csv based on German grammar rules.
German nouns should always be capitalized.
"""
import csv
import sys

# Words that should be lowercase (verbs, adjectives, adverbs, prepositions, etc.)
# This is not exhaustive but covers common patterns
LOWERCASE_PATTERNS = [
    'ab', 'aber', 'abfahren', 'abfliegen', 'abgeben', 'abholen', 'abschließen',
    'aktiv', 'aktuell', 'alle', 'allein', 'alles', 'als', 'also', 'alt',
    'an', 'anbieten', 'anderen', 'anders', 'anfangen', 'ankommen', 'anrufen',
    'antworten', 'arbeiten', 'auch', 'auf', 'aus', 'bei', 'bekannt', 'bestellen',
    'bis', 'bleiben', 'brauchen', 'bringen', 'da', 'dabei', 'dafür', 'dagegen',
    'dann', 'daran', 'darauf', 'darüber', 'darum', 'dass', 'dazu', 'dein',
    'dem', 'den', 'der', 'deshalb', 'deutlich', 'dich', 'die', 'dies', 'dir',
    'doch', 'dort', 'dran', 'drauf', 'drinnen', 'drüber', 'du', 'dumm', 'dunkel',
    'dünn', 'durch', 'dürfen', 'echt', 'ehren', 'eigen', 'eigentlich', 'ein',
    'einig', 'einpacken', 'eintragen', 'einverstanden', 'er', 'erkältet',
    'erlaubt', 'es', 'essen', 'euer', 'fahren', 'fallen', 'fertig', 'finden',
    'fit', 'fragen', 'frei', 'früher', 'fühlen', 'für', 'geben', 'geehrte',
    'gehen', 'gern', 'gestern', 'gültig', 'gut', 'haben', 'halten', 'hängen',
    'heiß', 'helfen', 'her', 'heute', 'hier', 'hin', 'hinein', 'hören',
    'ich', 'ihr', 'im', 'immer', 'in', 'interessieren', 'ist', 'ja', 'jeder',
    'jemand', 'jetzt', 'jung', 'kalt', 'kaufen', 'kennen', 'klein', 'kommen',
    'können', 'kümmern', 'kurz', 'lang', 'lassen', 'laufen', 'leben', 'legen',
    'lesen', 'lieben', 'liegen', 'machen', 'mal', 'man', 'mehr', 'mein',
    'meinen', 'mit', 'möchten', 'mögen', 'müssen', 'nach', 'nächste', 'nah',
    'nehmen', 'nein', 'neu', 'nicht', 'nichts', 'noch', 'nötig', 'nur',
    'ob', 'oben', 'oder', 'oft', 'ohne', 'passen', 'recht', 'richtig', 'rufen',
    'sagen', 'schauen', 'scheinen', 'schenken', 'schicken', 'schlafen', 'schlecht',
    'schließlich', 'schneiden', 'schnell', 'schön', 'schreiben', 'schwer', 'sehen',
    'sehr', 'sein', 'seit', 'setzen', 'sich', 'sie', 'sitzen', 'so', 'sollen',
    'sondern', 'spät', 'spazieren', 'spielen', 'sprechen', 'stehen', 'stellen',
    'streiten', 'tragen', 'treffen', 'trinken', 'tun', 'über', 'um', 'umziehen',
    'und', 'uns', 'unser', 'unter', 'unterhalten', 'unterwegs', 'verabredet',
    'verboten', 'vereinbaren', 'verletzen', 'verlieben', 'verstehen', 'viel',
    'vom', 'von', 'vor', 'vorstellen', 'wann', 'warm', 'warten', 'warum', 'was',
    'waschen', 'weg', 'wegen', 'weh', 'weiterhelfen', 'welcher', 'wenig', 'wenn',
    'wer', 'werden', 'wichtig', 'wie', 'wieder', 'windig', 'wir', 'wissen', 'wo',
    'wohnen', 'wollen', 'woher', 'wohin', 'zu', 'zurück', 'zurückfahren',
    'zurückgeben', 'zurückgehen', 'zurückkommen', 'zurücklaufen', 'zusammen',
    'ändern', 'ärgern', 'überweisen'
]

# Convert to set for faster lookup (case-insensitive)
LOWERCASE_SET = {word.lower() for word in LOWERCASE_PATTERNS}

def should_be_lowercase(word):
    """
    Determine if a word should be lowercase based on German grammar.
    Returns True if the word is a verb, adjective, adverb, etc.
    """
    word_lower = word.lower()
    
    # Check if in known lowercase set
    if word_lower in LOWERCASE_SET:
        return True
    
    # Compound words with hyphens containing reflexive verbs
    if 'sich-' in word_lower:
        return True
    
    # Words ending in verb-like patterns
    if word_lower.endswith(('-en', '-ern', '-ieren')):
        return True
    
    # Comparative/superlative adjectives
    if word_lower.endswith(('-er', '-ste', '-st')) and len(word) > 4:
        # But not nouns like "Fenster", "Vater", etc.
        base = word_lower[:-2] if word_lower.endswith('er') else word_lower[:-3]
        if base in LOWERCASE_SET:
            return True
    
    return False

def fix_capitalization(word):
    """
    Fix capitalization of a German word.
    Nouns should be capitalized, verbs/adjectives/etc should be lowercase.
    
    Note: Some words like "weg" can be both noun (Weg = path) and adverb (weg = away).
    These are typically distinguished by numbered variants (weg_1, weg_2) or compounds.
    The base form "weg" without suffix is assumed to be the noun form.
    """
    # Handle compound words with hyphens
    if '-' in word:
        # Keep original capitalization for complex compounds
        # unless it's a reflexive verb
        if word.lower().startswith('sich-'):
            return word.lower()
        # For other compounds, trust the original form
        return word
    
    # Handle numbered variants (e.g., "word_1")
    if '_' in word:
        parts = word.split('_')
        parts[0] = fix_capitalization(parts[0])
        return '_'.join(parts)
    
    if should_be_lowercase(word):
        return word.lower()
    else:
        # Capitalize first letter (noun)
        return word[0].upper() + word[1:].lower() if len(word) > 1 else word.upper()

def main():
    print("Reading A2.csv...")
    with open('A2.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Processing {len(rows)} words...")
    
    # Fix capitalization
    changed = 0
    for row in rows:
        original = row['German_Lemma']
        fixed = fix_capitalization(original)
        if original != fixed:
            row['German_Lemma'] = fixed
            changed += 1
            print(f"  Changed: {original} → {fixed}")
    
    print(f"\nFixed capitalization for {changed} words")
    
    # Write updated A2.csv
    print("Writing updated A2.csv...")
    with open('A2.csv', 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['German_Lemma', 'Spanish_Translation']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print("Done!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
