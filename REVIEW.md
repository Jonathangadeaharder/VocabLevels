# Trilingual CEFR Vocabulary Database — Code & Data Review

## Executive Summary
Database: **15 files, 24,000 vocabulary entries** (600/600/1000/2000/4000 per CEFR level × 3 languages)
Status: **Good structural integrity, critical bugs in tooling, quality gaps in data**

---

## 🔴 CRITICAL ISSUES

### 1. vocab_manager.py: `lookup` command broken (line 179)
**Issue**: Substring matching instead of exact match
```python
if any(needle == v.lower() or needle in v.lower() for v in values):
```
**Impact**: Searching for "s" returns 15,380 results (nearly all entries containing 's')
**Fix**: Remove substring part, use exact match only
```python
if any(needle == v.lower() for v in values):
```

### 2. Plural and verb-form duplication in data
**Issue**: Both singular and plural exist as separate entries
- English: "cat" (A2) + "cats" (B1)
- Violates CEFR lemmatization principle: entries should be dictionary forms (lemmas), not inflected forms

**Impact**: ~877 entries are inflected forms, increasing database size by 15% without semantic value
**Root cause**: Data sourced from frequency lists (raw text) not properly lemmatized during consolidation

---

## 🟡 HIGH PRIORITY ISSUES

### 3. check_quality.py: Insufficient validation (line 70-104)
Missing checks:
- **Empty translations after strip**: `"  "` (spaces only) passes validation
- **Translation quality**: No check if German/Spanish translations are plausible (e.g., detecting placeholder text)
- **Case consistency**: Lemmas can start with capital (e.g., "Berlin"), should normalize to lowercase
- **Accented character handling**: Spanish/German accents sometimes inconsistent
- **Duplicate translations within level**: Two lemmas with identical German translation could indicate copy-paste error

### 4. vocab_manager.py: Input validation gaps
- `cmd_update --rename`: No check if new lemma already exists (creates silent duplicate)
- `cmd_add`: Doesn't strip whitespace from input (user types " cat " → stored as " cat ")
- `cmd_move`: Doesn't validate target_level before processing
- Line 140: Calls `cmd_remove` as side-effect, poor separation of concerns

### 5. Transaction safety missing
No atomic operations: if `write_level` fails mid-write, CSV is corrupted
No backup: single write error = data loss

---

## 🟢 WORKING WELL

✅ **check_quality.py**: Header validation, cross-level deduplication, special char detection
✅ **vocab_manager.py**: find/add/remove/move/update commands work for normal use
✅ **Data structure**: Alphabetical sorting, consistent schema across languages
✅ **Coverage**: All levels hit exact targets (600/600/1000/2000/4000)

---

## DETAILED FINDINGS

### Data Quality

| Check | Status | Details |
|-------|--------|---------|
| Schema consistency | ✅ | All 15 files have correct headers |
| Empty lemmas | ✅ | 0 found |
| Multi-word lemmas | ✅ | 0 found |
| Special chars in lemmas | ✅ | 0 found |
| Intra-level duplicates | ✅ | 0 found |
| Cross-level duplicates | ✅ | 0 found |
| **Inflected forms** | ❌ | ~877 (plurals, verb forms, etc.) |
| **Empty/whitespace translations** | ❓ | Not checked |
| **Translation plausibility** | ❌ | Not validated |

### Code Quality

#### check_quality.py
- **Strengths**: Clear, focused validation logic; good error reporting
- **Weaknesses**: Only validates structure, not content; no severity levels
- **Coverage**: 70% of needed checks

#### vocab_manager.py
- **Strengths**: Works for common operations; good CLI structure
- **Weaknesses**: Substring search bug; no input validation; no atomicity
- **Coverage**: 60% of needed features (missing batch operations, export, statistics)

---

## 🔧 RECOMMENDED FIXES (Priority Order)

### Phase 1: Critical (Day 1)
1. **Fix lookup command** (vocab_manager.py:179)
   - Change `needle in v.lower()` → remove substring matching
   - Add test: `lookup "cat"` should NOT return "cats" or "locate"

2. **Add rename validation** (vocab_manager.py:156-161)
   - Check if renamed lemma already exists in target language
   - Prevent silent overwrite

3. **Add input normalization** (vocab_manager.py:85-102)
   - Strip whitespace: `args.lemma.strip()`
   - Apply to add, remove, update, move commands

### Phase 2: High (Week 1)
4. **Lemmatization audit**
   - Create script to identify inflected forms
   - Decide: keep plurals/verb-forms, or lemmatize to base forms
   - Remove duplicates if lemmatizing

5. **Enhance check_quality.py**
   - Add whitespace-only translation detection
   - Add case consistency check (lemmas should be lowercase)
   - Add option to fix issues automatically

6. **Add transaction safety**
   - Use temp file + rename pattern (atomic writes)
   - Add backup before write in vocab_manager operations

### Phase 3: Nice-to-have (Later)
7. **Add statistics command**: `vocab_manager.py stats [lang]`
8. **Add batch operations**: `vocab_manager.py import german data.csv`
9. **Add export**: `vocab_manager.py export spanish B1 --format json`
10. **Add cross-language consistency check**: Verify translation pairs are reciprocal

---

## FILES AFFECTED

- ✏️ **vocab_manager.py** — fix lookup, add validation
- ✏️ **check_quality.py** — add content validation
- 📊 **All CSV files** — need lemmatization audit
- 📝 **README.md** — document schema, constraints, known issues

---

## TIMELINE
- **Phase 1** (critical fixes): 1-2 hours
- **Phase 2** (quality improvements): 4-6 hours
- **Phase 3** (features): 6-8 hours
