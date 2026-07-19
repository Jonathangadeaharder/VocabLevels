# TNG CEFR audit samples (p20) — REVIEWED

Design: **95% CI, ±20%**, p=0.5, FPC. Seed EN `20260717`; others `20260718`.

All packs scored: `verdict` in each `*.sample-p20.csv` + MD tables.

## Results

| Lang | n | keep | fix | drop | defect rate |
|------|--:|-----:|----:|-----:|------------:|
| english | 120 | 117 | 3 | 0 | 2.5% |
| french | 118 | 115 | 3 | 0 | 2.5% |
| dutch | 118 | 113 | 5 | 0 | 4.2% |
| german | 95 | 88 | 7 | 0 | 7.4% |
| spanish | 96 | 94 | 2 | 0 | 2.1% |
| swedish | 94 | 91 | 2 | 1 | 3.2% |
| arabic | 72 | 67 | 5 | 0 | 6.9% |
| chinese | 24 | 24 | 0 | 0 | 0.0% |

## Defect log (non-keep)

### english

- **A2** `dreams` → **fix**: Lemma is plural 'dreams'; citation form must be 'dream'. english_lemma already dream; incomplete fix.
- **B1** `cells` → **fix**: Lemma is plural 'cells'; citation form must be 'cell'. english_lemma already cell.
- **B1** `answered` → **fix**: Lemma is past tense 'answered'; citation form must be 'answer'. POS VERB ok; english_lemma already answer.
### french

- **B1** `autoentrepreneur` → **fix**: Standard orthography is auto-entrepreneur (hyphen).
- **B1** `intituler` → **fix**: english_lemma 'entitle' is wrong sense; use 'to title / to be titled'.
- **C1** `travers` → **fix**: Bare 'travers' is a NOUN; preposition is 'à travers'. Either lemma=à travers (ADP) or travers/NOUN.
### dutch

- **A1** `mogen` → **fix**: UPOS VERB; modal ability/permission is usually AUX (cf. A2 sample mogen/AUX). Align UPOS to AUX.
- **B1** `honderden` → **fix**: Plural 'hundreds'; prefer lemma honderd (NUM) or treat as NOUN carefully.
- **B2** `vak` → **fix**: english_lemma 'compartment' is niche; common CEFR senses are subject/field or tray. Prefer 'subject / field'.
- **C1** `aanvoeren` → **fix**: english_lemma 'to submit' is weak/secondary; primary senses lead / put forward / command.
- **C1** `uitdagingen` → **fix**: Plural form; citation lemma must be uitdaging.
### german

- **A1** `ander` → **fix**: UPOS DET is wrong; citation form is andere/ADJ (or ander- stem). Prefer andere/ADJ.
- **C1** `Kooperation` → **fix**: chinese_lemma still 'Kooperation' (not Chinese); set 合作.
- **C1** `Besonderes` → **fix**: Nominalized neuter; prefer besonder/ADJ or drop as non-lemma.
- **C1** `Heiliger` → **fix**: Inflected noun; citation Heiligen is wrong too — prefer Heilige/NOUN or heilig/ADJ.
- **C1** `meinten` → **fix**: Finite past of meinen; citation lemma must be meinen.
- **C1** `Krachen` → **fix**: Not standard citation: use Krach/NOUN or krachen/VERB.
- **C1** `Schwarzer` → **fix**: Inflected noun form; prefer schwarz/ADJ. Sensitive headword — review inclusion policy.
### spanish

- **B1** `ordenar` → **fix**: english_lemma only 'to order' misses primary 'to tidy/organize'; Chinese 整理 is better than 订购 sense.
- **C1** `émulo` → **fix**: english_lemma 'emulator' is tech false friend; means rival/emulator (person). Use 'rival'.
### swedish

- **A1** `andra` → **fix**: Form is plural/determiner 'others'; lemma often annan/ADJ or andra as DET — set UPOS DET or lemma annan.
- **B1** `unga` → **fix**: Plural/weak form of ung; citation lemma ung/ADJ.
- **B2** `los` → **drop**: Not standard Swedish for 'loose' (that is lös). Looks like noise/loan fragment.
### arabic

- **A1** `حس` → **fix**: english_lemma 'sense' OK as noun; was feel/felt — verify POS NOUN vs VERB أحس.
- **A1** `واعر` → **fix**: Colloquial Maghrebi/Levantine slang; flag for MSA CEFR policy (keep only if dialect allowed).
- **A2** `تقلق` → **fix**: Conjugated 2sg form; use bare lemma قلق as VERB/NOUN citation form.
- **A2** `نشر` → **fix**: english_lemma only 'hang laundry' is too narrow; also publish/spread. Broaden gloss.
- **B1** `فَايِق` → **fix**: Dialectal (awake); MSA would be مستيقظ. Policy flag for dialect.

## Paths

`manual_reviews/{lang}/tng-audit-sample/`

Regenerate samples: `uv run python -m scripts.gemma_qa.build_audit_sample --root .`


## Applied (2026-07-19)

All defect-log fix/drop items and skeptic gaps applied to committed CSVs:
meinten→meinen, bräuchten→brauchen, uitdagingen→uitdaging, honderden→honderd,
los dropped, émulo→rival, تقلق→قلق, auto-entrepreneur, intituler gloss,
andere, besonder, Heilige, Krach, schwarz, Marge ZH, zullen shall/will, etc.
Post-fix seed 20260719:postfix3: 951 keep / 0 defects.
