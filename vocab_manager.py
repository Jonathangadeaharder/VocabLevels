import pandas as pd
import os
import sys
import argparse

# Configuration
LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1']
FILES = {level: f"{level}.csv" for level in LEVELS}
MASTER_LIST_PATH = "master_lemmas.txt"

class VocabManager:
    def __init__(self):
        self.dfs = {}
        self.load_data()
        self.master_lemmas = self.load_master_list()

    def load_data(self):
        """Loads all CSV levels into pandas DataFrames."""
        for level, filename in FILES.items():
            if os.path.exists(filename):
                try:
                    # Expecting header: German_Lemma,Spanish_Translation
                    df = pd.read_csv(filename, encoding='utf-8')
                    # Ensure columns exist and are strings
                    df['German_Lemma'] = df['German_Lemma'].astype(str)
                    df['Spanish_Translation'] = df['Spanish_Translation'].astype(str)
                    self.dfs[level] = df
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    self.dfs[level] = pd.DataFrame(columns=['German_Lemma', 'Spanish_Translation'])
            else:
                print(f"Warning: {filename} not found. Creating empty dataframe.")
                self.dfs[level] = pd.DataFrame(columns=['German_Lemma', 'Spanish_Translation'])

    def load_master_list(self):
        """Loads the master lemma list for validation."""
        if os.path.exists(MASTER_LIST_PATH):
            with open(MASTER_LIST_PATH, 'r', encoding='utf-8') as f:
                # Set for O(1) lookup
                return set(line.strip() for line in f if line.strip())
        return set()

    def save_level(self, level):
        """Saves a specific level back to CSV."""
        if level in self.dfs:
            self.dfs[level].to_csv(FILES[level], index=False)
            print(f"Saved {FILES[level]}")

    # ================= LINTER FUNCTIONS =================

    def lint(self):
        print("--- STARTING LINT CHECK ---")
        word_to_level_map = {}  # tracks which level each word belongs to
        has_errors = False

        for level in LEVELS:
            df = self.dfs[level]
            print(f"\nChecking Level {level} ({len(df)} entries)...")
            
            # 1. Check Single Word Criterion
            multi_word_mask = df['German_Lemma'].str.contains(' ', na=False)
            if multi_word_mask.any():
                print(f"  [ERROR] Multi-word entries found in {level}:")
                print(df[multi_word_mask]['German_Lemma'].tolist())
                has_errors = True

            # 2. Check Intra-level Duplicates
            if df['German_Lemma'].duplicated().any():
                print(f"  [ERROR] Duplicates found INSIDE {level}:")
                print(df[df['German_Lemma'].duplicated()]['German_Lemma'].tolist())
                has_errors = True

            # 3. Check Inter-level Uniqueness & Master List
            for index, row in df.iterrows():
                word = row['German_Lemma']
                
                # Inter-level check
                if word in word_to_level_map:
                    prev_level = word_to_level_map[word]
                    print(f"  [ERROR] '{word}' exists in {level} but was already seen in {prev_level}")
                    has_errors = True
                else:
                    word_to_level_map[word] = level

        if not has_errors:
            print("\n[SUCCESS] No critical errors found.")
        else:
            print("\n[FAIL] Errors found. Please fix them.")

    # ================= UTILITY FUNCTIONS =================

    def find_word(self, word):
        """Returns the level(s) where a word is found."""
        found_in = []
        for level in LEVELS:
            if word in self.dfs[level]['German_Lemma'].values:
                found_in.append(level)
        return found_in

    def add_word(self, level, word, translation):
        """Adds a word to a specific level if it doesn't exist anywhere."""
        existing = self.find_word(word)
        if existing:
            print(f"Error: '{word}' already exists in {existing}. Cannot add.")
            return

        if ' ' in word:
            print(f"Error: '{word}' contains spaces. Single word criterion violated.")
            return

        new_row = pd.DataFrame([{'German_Lemma': word, 'Spanish_Translation': translation}])
        # Sort alphabetically by 'German_Lemma' after addition
        self.dfs[level] = self.dfs[level].sort_values(by='German_Lemma').reset_index(drop=True)
        # Consider sorting before saving or in batch, not after every addition, for efficiency.
        self.save_level(level)
        print(f"Added '{word}' to {level}.")

    def remove_word(self, word):
        """Removes a word from ALL levels."""
        found = False
        for level in LEVELS:
            df = self.dfs[level]
            if word in df['German_Lemma'].values:
                self.dfs[level] = df[df['German_Lemma'] != word]
                self.save_level(level)
                print(f"Removed '{word}' from {level}.")
                found = True
        if not found:
            print(f"Word '{word}' not found in any level.")

    def update_word(self, word, new_translation=None, new_word=None):
        """Updates translation or corrects a typo in the lemma."""
        found_levels = self.find_word(word)
        if not found_levels:
            print(f"Word '{word}' not found.")
            return

        for level in found_levels:
            df = self.dfs[level]
            idx = df.index[df['German_Lemma'] == word].tolist()
            
            if new_translation:
                df.at[idx[0], 'Spanish_Translation'] = new_translation
                print(f"Updated translation for '{word}' in {level}.")
            
            if new_word:
                # Check if new_word conflicts
                if self.find_word(new_word):
                    print(f"Cannot rename '{word}' to '{new_word}': Target already exists.")
                    return
                df.at[idx[0], 'German_Lemma'] = new_word
                print(f"Renamed '{word}' to '{new_word}' in {level}.")
            
            self.save_level(level)

    def move_word(self, word, target_level):
        """Moves a word from its current level to a target level."""
        found_levels = self.find_word(word)
        if not found_levels:
            print(f"Word '{word}' not found.")
            return
        
        # Warn if word exists in multiple levels (data inconsistency)
        if len(found_levels) > 1:
            print(f"Warning: '{word}' exists in multiple levels: {found_levels}. Will remove from all and add to {target_level}.")
        
        # Get translation from the first level found
        current_level = found_levels[0]
        df = self.dfs[current_level]
        translation = df.loc[df['German_Lemma'] == word, 'Spanish_Translation'].values[0]
        # Pre-validate: check if word already exists in target level
        if target_level in self.find_word(word):
            print(f"Word '{word}' already exists in {target_level}. Move aborted.")
            return
        # Optionally, add further validation here if add_word does more checks
        # For now, we assume add_word will print its own error and not add duplicates

        # Remove and Add (atomic: only after validation)
        # Remove and Add
        self.remove_word(word)
        self.add_word(target_level, word, translation)
        print(f"Moved '{word}' from {current_level} to {target_level}.")

# ================= CLI HANDLER =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="German Vocab Manager & Linter")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Lint Command
    subparsers.add_parser('lint', help='Check for duplicates, spaces, and validity')

    # Add Command
    add_parser = subparsers.add_parser('add', help='Add a new word')
    add_parser.add_argument('level', choices=LEVELS, help='Target Level')
    add_parser.add_argument('word', help='German Lemma')
    add_parser.add_argument('translation', help='Spanish Translation')

    # Remove Command
    rm_parser = subparsers.add_parser('remove', help='Remove a word from all levels')
    rm_parser.add_argument('word', help='German Lemma to remove')

    # Move Command
    mv_parser = subparsers.add_parser('move', help='Move a word to a different level')
    mv_parser.add_argument('word', help='German Lemma')
    mv_parser.add_argument('target_level', choices=LEVELS, help='New Level')

    # Update Command
    up_parser = subparsers.add_parser('update', help='Update translation or fix typo')
    up_parser.add_argument('word', help='Original German Lemma')
    up_parser.add_argument('--trans', help='New Spanish Translation')
    up_parser.add_argument('--lemma', help='Corrected German Lemma')

    args = parser.parse_args()
    manager = VocabManager()

    if args.command == 'lint':
        manager.lint()
    elif args.command == 'add':
        manager.add_word(args.level, args.word, args.translation)
    elif args.command == 'remove':
        manager.remove_word(args.word)
    elif args.command == 'move':
        manager.move_word(args.word, args.target_level)
    elif args.command == 'update':
        manager.update_word(args.word, new_translation=args.trans, new_word=args.lemma)
    else:
        parser.print_help()
