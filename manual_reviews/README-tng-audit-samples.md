# TNG CEFR audit samples (p20)

Stratified random samples for **succeeded** CEFR scale levels.

- Design: **95% CI, ±20% margin**, p=0.5, finite-population corrected
- Per level typically **n ≈ 23–24**
- Seed: `20260718` (English pack: `20260717`, already scored)

## Packs

| Lang | Path | Levels | Status |
|------|------|--------|--------|
| en | `english/tng-audit-sample/` | A1–C1 | **REVIEWED** |
| fr | `french/tng-audit-sample/` | A1–C1 | UNREVIEWED |
| nl | `dutch/tng-audit-sample/` | A1–C1 | UNREVIEWED |
| de | `german/tng-audit-sample/` | A1 A2 B1 C1 | UNREVIEWED |
| es | `spanish/tng-audit-sample/` | A1 A2 B1 C1 | UNREVIEWED |
| sv | `swedish/tng-audit-sample/` | A1–B2 | UNREVIEWED |
| ar | `arabic/tng-audit-sample/` | A1–B1 | UNREVIEWED |
| zh | `chinese/tng-audit-sample/` | A1 | UNREVIEWED |

Regenerate after more scale successes:

```bash
uv run python -m scripts.gemma_qa.build_audit_sample --root .
```

Score: fill `verdict` in `*.sample-p20.csv` / tables in `*.md` (`keep` / `drop` / `fix`).
