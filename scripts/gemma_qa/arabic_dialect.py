"""Full-population Arabic dialect / colloquial / French-loan classifier.

Pure functions: classify every lemma against closed token sets + patterns.
Inventory is the output of scan_arabic_lists(), not a skeptic hit-list.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

LEVELS: tuple[str, ...] = ("A1", "A2", "B1", "B2", "C1")


def strip_ar_diacritics(value: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFC", value)
        if not ("\u064b" <= c <= "\u065f") and c not in "\u0670"
    )


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value or "").strip()


# ---------------------------------------------------------------------------
# Closed token sets (Maghrebi / Egyptian / Levantine colloquial + FR loans)
# ---------------------------------------------------------------------------

# Discourse / function (Darija + Egyptian)
_FUNC: frozenset[str] = frozenset(
    {
        "فوقاش",
        "فوقتاش",
        "بصح",
        "غادي",
        "غادين",
        "غادة",
        "حيت",
        "باش",
        "واخا",
        "داكشي",
        "داك",
        "دوك",
        "بزاف",
        "كاين",
        "كاينة",
        "كاينين",
        "ماشي",
        "فين",
        "علاش",
        "كيفاش",
        "منين",
        "اش",
        "شنو",
        "اشنو",
        "واش",
        "هادشي",
        "هاد",
        "هادا",
        "هادي",
        "هادو",
        "نيت",
        "يالله",
        "يلاه",
        "دراري",
        "هاك",
        "هاكا",
        "ياك",
        "إيوا",
        "ايوا",
        "إييه",  # Maghrebi yes
        "اييه",

        "ديال",
        "ديالي",
        "ديالك",
        "ديالو",
        "ديالها",
        "راه",
        "راني",
        "راهي",
        "راهم",
        "راكم",
        "بلاتي",
        "مبروك",
        "بنة",
        "طاح",
        "سير",
        "صافي",
        "شكون",
        "شحال",
        "بغيت",
        "نبغي",
        "ندير",
        "نمشيو",
        "لاباس",
        "دابا",
        "دبا",
        "شوية",
        "بشوية",
        "عافاك",
        "عفاك",
        "خويا",
        "ختي",
        "واقيلا",
        "زعما",  # Maghrebi/colloquial "supposedly"
        "إمتا",  # Maghrebi when (MSA متى)
        "امتا",
        "دغيا",  # Maghrebi quickly
        "بزربة",  # Maghrebi in a hurry
        "يعني",  # only with PART handled in classify
        "كون",  # conditional only
        "لازم",  # AUX only
    }
)

# Maghrebi numerals (not MSA ثلاثة / اثنان / …)
_NUMERALS: frozenset[str] = frozenset(
    {
        "جوج",
        "زوج",
        "تلاتة",
        "تلاته",
        "تلاتين",
        "تمنية",
        "تسعود",
        "حداش",
        "حدش",  # eleven (Maghrebi)
        "طناش",
        "تلتاش",
        "ربعطاش",
        "خمسطاش",
        "سطاش",
        "سبعطاش",
        "تمنتاش",
        "تسعطاش",
        "تمانية",  # eight (Maghrebi; MSA ثمانية)
        "تمانين",  # eighty
        "تمنطاش",  # eighteen
        "تمنطاش",
        "تسعود",
    }
)

# French / Romance loans common in Maghrebi speech (not formal MSA)
_LOANS: frozenset[str] = frozenset(
    {
        "بارطما",
        "برطما",
        "كراج",
        "دوش",
        "كراء",
        "كرى",
        "كرا",
        "بوليس",
        "بوليسية",
        "كبطان",
        "كبتان",
        "طوموبيل",
        "تليفون",
        "فاميلا",
        "كوزينة",
        "شومبر",
        "فريجيدر",
        "سبيطار",
        "مارشي",
        "كارطة",
        "تابلة",
        "فاليز",
        "باسبور",
        "تران",
        "بروفيل",
        "فيرمة",
        "شيشة",
        "بيسة",
        "سالون",
        "بانيو",  # Romance loan bathtub (not formal MSA حمام)
        "صندويش",
        "ساندويش",
        "اتوبيس",
        "كنبة",
        "تيليفون",
        "شوفاج",  # FR chauffage (heater) — Maghrebi
        "دبلوم",  # FR diplôme (MSA شهادة)
        "دبلومة",
        "دِبْلوم",
        "دَبْلوم",
        "تاكسي",  # taxi loan
        "طاكسي",
        "كونيكسيون",  # FR connexion
        "كونكسيون",
        "مايكة",  # Maghrebi plastic bag (MSA كيس)
        "مايكا",
        "جمركة",  # Maghrebi admin for customs clearance (MSA تخليص جمركي)
        "سنتيم",  # FR centime
        "ستوري",  # EN story (social)
        "فرملة",  # FR frein brake
        "تلفزة",  # Maghrebi television
        "تلفازة",
        "ترام",  # tram loan
        "ستاج",  # FR stage internship
        "ستاجير",
        "مارطو",  # FR marteau hammer (MSA مطرقة)
        "ريبرتوار",  # FR répertoire
        "صالون",  # FR salon living room
        "روبيني",  # FR robinet faucet
        "بيجامة",  # FR pyjama
        "بيجاما",
        "كالامار",  # FR calamar squid
        "ترامواي",  # FR tramway
        "ديبلوم",  # FR diplôme variant
        "شبشب",  # colloquial slippers (MSA خف)
        "كابتن",  # EN captain (MSA قبطان)
        "مازوت",  # FR mazout diesel
        "لوجستيك",  # FR logistique
        "أتاي",  # Maghrebi tea (MSA شاي)
        "اتاي",
        "كورجيت",  # FR courgette
        "كُورْجِيت",
        "ريجيم",  # FR régime diet
    }
)

# Dialect adjectives / stems
_ADJ: frozenset[str] = frozenset(
    {
        "جيعان",
        "جيعانة",
        "عطشان",  # also MSA; keep via MSA allow below if needed — treat dialect when tagged hungry-only
        "ناعس",
        "نعسان",
        "نعسانة",
        "زوين",
        "زوينة",
        "مزيان",
        "مزيانة",
        "خايب",
        "خايبة",
        "واعر",
        "واعرة",
        "زعفان",  # Maghrebi angry (MSA غاضب)
        "زعفانة",
        "خايف",  # colloquial afraid (MSA خائف)
        "خايفة",
        "خايفين",
        "خدوم",  # Maghrebi intensive "helpful"
        "خَدُوم",
        "خدوما",
        "مرمد",
        "مرمدة",
        "خاوي",  # Maghrebi empty (MSA فارغ/خاوٍ)
        "خاوية",
        "مقرقب",  # Maghrebi crunchy
        "مُقَرْقَب",
        "مسوس",  # Maghrebi tasteless
        "مَسُوس",
        "هانئة",  # Maghrebi fine/no problem (discourse)
        "هانئ",
        "ناشف",  # Maghrebi dry (MSA جاف)
        "ناشفة",
        "مزروب",  # Maghrebi in a hurry
        "مَزْرُوب",
        "محشوم",  # Maghrebi modest
        "مَحْشُوم",
        "مبلول",  # Maghrebi wet
        "مَبْلُول",
        "تقيل",  # Maghrebi heavy (MSA ثقيل)
        "بردان",  # Maghrebi feeling cold
        "بردانة",
        "زاهي",  # Maghrebi vivid/bright
    }
)

# Colloquial verbs (Egyptian / Levantine / Maghrebi citation forms)
_VERBS: frozenset[str] = frozenset(
    {
        "جاب",
        "جابو",
        "جيبي",
        "كمل",
        "كملي",
        "كملوا",
        "تغدى",
        "تغديت",
        "اتغدى",
        "اتفضل",
        "اتفضلي",
        "شوف",
        "شوفي",
        "شوفوا",
        "هات",
        "هاتي",
        # Egyptian imperative "go" — only with non-NOUN UPOS (روح NOUN = MSA soul)
        "روحي",
        "يلا",
        "خلاص",
        "اوكي",
        "توحش",  # Maghrebi "to miss someone" (وحشتني)
        "وحشت",
        "وحشتني",
        "عاود",  # Maghrebi "repeat" (MSA أعاد)
        "عاودت",
        "عاودي",
        "عاودوا",
        "زعف",  # Maghrebi "get angry"
        "تزعف",
        "مخط",  # Maghrebi "blow the nose" (MSA تمخط)
        "تمخط",
        "خربق",  # Maghrebi mess up
        "خربقة",
        "لزق",  # Maghrebi stick (MSA لصق)
        "جاوب",  # Maghrebi answer (MSA أجاب)
        "جاوبي",
        "جاوبوا",
        "وضب",  # Maghrebi tidy
        "وَضَّبَ",
        "وضبت",
        "دلل",  # Maghrebi spoil a child
        "خوى",  # Maghrebi empty
        "بغى",  # Maghrebi want (MSA أراد)
        "بغي",
        "تصنط",  # Maghrebi eavesdrop
        "تَصَنَّطَ",
        "شاف",  # Maghrebi see (MSA رأى)
        "شوف",  # already may exist
    }
)
_VERB_ONLY: frozenset[str] = frozenset({"روح"})  # drop only if not NOUN

# Maghrebi nouns
_NOUNS: frozenset[str] = frozenset(
    {
        "ميدة",
        "ميدات",
        "كرعين",
        "كساب",
        "تداريب",  # often Maghrebi plural of تدريب; MSA تدريبات
        "مناكش",
        "منقاش",
        "دري",
        "درية",
        "قدّام",
        "قدام",
        "ورا",
        "فوقاني",
        "تحتاني",
        "جلبانة",  # Maghrebi peas (MSA بازلاء)
        "جلبان",
        "قمرون",  # Maghrebi shrimp (MSA روبيان/جمبري)
        "قمرونة",
        "ضو",  # Maghrebi light (MSA ضوء)
        "تقشاب",  # Maghrebi joking
        "تقشابة",
        "شباكية",  # Maghrebi pastry
        "مسمن",
        "مُسَمَّن",
        "ملوي",
        "حرشة",
        "بغرير",
        "حيط",  # Maghrebi wall (MSA حائط)
        "بوز",  # FR/Maghrebi buzz (not MSA)
        "تقاشر",  # Maghrebi socks (MSA جوارب)
        "كولة",  # Maghrebi glue (MSA صمغ)
        "آزير",  # Maghrebi rosemary
        "ازير",
        "كرموس",  # Maghrebi fig
        "هندية",  # Maghrebi prickly pear (when eng=prickly pear — still drop token)
        "نفار",  # Maghrebi festival trumpet
        "جبصية",  # Maghrebi carved plaster (MSA جص)
        "بالي",  # Maghrebi worn out
        "برمة",  # Maghrebi cooking pot
        "ريحة",  # Maghrebi smell (MSA رائحة)
        "دروج",  # Maghrebi stairs (MSA درج)
        "طنجية",  # Maghrebi stew dish
        "فقاص",  # Maghrebi cookie
        "تحيين",  # Maghrebi updating (MSA تحديث)
        "طيفور",  # Maghrebi low round table
        "كيسان",  # Maghrebi tea glasses (dual sense)
    }
)

# Explicit MSA allowlist when token also appears dialectally
_MSA_KEEP: frozenset[str] = frozenset(
    {
        "فم",
        "كلّي",
        "كلي",
        "عامية",
        "لهجة",
        "جدلية",
        "فصحى",
        "عطشان",  # MSA thirsty — keep
        "خمسة",
        "ستة",
        "سبعة",
        "ثمانية",
        "تسعة",
        "عشرة",
        "ربع",
        "ثلاثة",
        "ثلاثون",
        "اثنان",
        "جائع",
        "يعني",  # MSA VERB to mean — keep as VERB only
        "أمام",
        "خلف",
        "تدريب",
        "تدريبات",
        "ملف",
        "طاولة",
        "مائدة",
        "روح",  # MSA soul/spirit (NOUN)
        "بخير",  # MSA fine/well
        "فيلا",  # international loan villa — accept in CEFR
        "نفس",
        "خير",
    }
)

_TOKEN_DROP: frozenset[str] = (
    _FUNC | _NUMERALS | _LOANS | _ADJ | _VERBS | _NOUNS
) - _MSA_KEEP

# Pattern rules (applied after allowlist)
_LOAN_SUFFIX_RE = re.compile(
    r"^(?!.{0,2}$)[\u0600-\u06FF]{2,}(اج|يش|يون|يير|وار|ورة|اتور|سيون|مون)$"
)
_MAGHREBI_PREFIX_RE = re.compile(r"^(ب|ف|ك)(?!ال)[\u0600-\u06FF]{2,}$")  # b-/f-/k- clitic forms like بشوية
# Known clitic+dialect stems
_CLITIC_STEMS: frozenset[str] = frozenset(
    {
        "شوية",
        "زاف",
        "صح",
        "لاص",
        "لاصة",
        "خير",
        "حال",
    }
)


@dataclass(frozen=True)
class ClassifyResult:
    action: str  # drop | policy | ok
    reason: str


def _strip_al(bare: str) -> str:
    """Strip Arabic definite article ال for closed-set match (الكساب → كساب)."""
    if bare.startswith("ال") and len(bare) > 3:
        return bare[2:]
    return bare


def classify_ar_lemma(
    lemma: str,
    upos: str = "",
    english: str = "",
) -> ClassifyResult:
    """Return drop|policy|ok for one Arabic headword."""
    lem = _nfc(lemma)
    if not lem:
        return ClassifyResult("ok", "empty")
    bare = strip_ar_diacritics(lem)
    stem = _strip_al(bare)
    en = (english or "").strip().lower()
    up = (upos or "").strip().upper()
    drop_set = {strip_ar_diacritics(x) for x in _TOKEN_DROP}
    keep_set = {strip_ar_diacritics(x) for x in _MSA_KEEP}

    # UPOS-conditional drops first (before MSA allow short-circuit).
    if lem == "يعني" and up == "PART":
        return ClassifyResult("drop", "يعني discourse PART colloquial")
    if lem == "كون" and (up in {"SCONJ", "CCONJ", "PART"} or en in {"if", "if only"}):
        return ClassifyResult("drop", "Darija conditional كون")
    if lem == "لازم" and up == "AUX":
        return ClassifyResult("drop", "لازم AUX Maghrebi")
    if (lem in _VERB_ONLY or bare in _VERB_ONLY or stem in _VERB_ONLY) and up not in {
        "NOUN",
        "",
    }:
        return ClassifyResult("drop", "colloquial imperative روح")

    if lem in _MSA_KEEP or bare in keep_set or stem in keep_set:
        return ClassifyResult("ok", "msa_allow")

    if lem in _TOKEN_DROP or bare in drop_set or stem in drop_set:
        return ClassifyResult("drop", "closed dialect/loan token set")

    # b-/f- clitic on dialect stem (بشوية) — not بخير (MSA allow)
    if (
        lem not in _MSA_KEEP
        and len(lem) >= 3
        and lem[0] in "بفك"
        and strip_ar_diacritics(lem[1:])
        in {strip_ar_diacritics(x) for x in _CLITIC_STEMS | _TOKEN_DROP}
    ):
        return ClassifyResult("drop", "clitic+dialect stem")

    # French-loan morphology (بروفيل already in set; general pattern)
    if re.fullmatch(r"[\u0600-\u06FF]{3,}(يل|اج|يش|يون)$", bare):
        # high false-positive risk — only if eng looks loan
        if any(
            k in en
            for k in (
                "profile",
                "garage",
                "police",
                "apartment",
                "shower",
                "rent",
                "captain",
                "coat",
                "car",
                "phone",
                "train",
                "passport",
            )
        ):
            return ClassifyResult("drop", "French-loan morphology+eng")

    if "colloquial" in en or "dialect" in en or "عامي" in en:
        if lem not in _MSA_KEEP:
            return ClassifyResult("policy", "eng labels colloquial/dialect")

    # Ambiguous MSA surface with Maghrebi sense in eng gloss.
    if bare == "سحاب" and "zipper" in en:
        return ClassifyResult("drop", "Maghrebi zipper (MSA سحاب=cloud)")
    if bare == "عمر" and en.strip() in {"never", "not ever"}:
        return ClassifyResult("drop", "Maghrebi never (MSA أبداً)")
    if bare in {"معمر", "مُعَمَّر"} and en.strip() in {"full", "stuffed"}:
        return ClassifyResult("drop", "Maghrebi full/sated")
    # Maghrebi "or" (MSA أو); MSA ولا is "and not"/nor, not plain "or".
    if bare == "ولا" and (
        en.strip() == "or" or (up in {"CCONJ", "CONJ"} and "or" in en and "not" not in en)
    ):
        return ClassifyResult("drop", "Maghrebi or (MSA أو)")

    return ClassifyResult("ok", "msa_or_unmarked")


@dataclass(frozen=True)
class InventoryRow:
    lang: str
    level: str
    lemma: str
    english_lemma: str
    chinese_lemma: str
    upos: str
    action: str
    reason: str


def scan_arabic_lists(root: Path) -> list[InventoryRow]:
    """Full-population scan of arabic/{A1–C1}.csv → inventory rows."""
    out: list[InventoryRow] = []
    for level in LEVELS:
        path = root / "arabic" / f"{level}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            lk = fields[0]
            for row in reader:
                lemma = _nfc(row.get(lk) or "")
                en = _nfc(row.get("English_Lemma") or "")
                zh = _nfc(row.get("Chinese_Lemma") or "")
                upos = (row.get("POS") or "").strip()
                result = classify_ar_lemma(lemma, upos=upos, english=en)
                if result.action == "ok":
                    continue
                out.append(
                    InventoryRow(
                        lang="ar",
                        level=level,
                        lemma=lemma,
                        english_lemma=en,
                        chinese_lemma=zh,
                        upos=upos,
                        action=result.action,
                        reason=result.reason,
                    )
                )
    # unique by level+lemma+upos+action
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[InventoryRow] = []
    for row in out:
        key = (row.level, row.lemma, row.upos, row.action)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def write_inventory(path: Path, rows: Sequence[InventoryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "lang",
                "level",
                "lemma",
                "english_lemma",
                "chinese_lemma",
                "upos",
                "action",
                "reason",
            ],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r.level, r.lemma)):
            writer.writerow(
                {
                    "lang": row.lang,
                    "level": row.level,
                    "lemma": row.lemma,
                    "english_lemma": row.english_lemma,
                    "chinese_lemma": row.chinese_lemma,
                    "upos": row.upos,
                    "action": row.action,
                    "reason": row.reason,
                }
            )


def load_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def closed_lexicon_inventory() -> list[InventoryRow]:
    """Frozen drop lexicon from classifier closed sets (not live list hits)."""
    rows: list[InventoryRow] = []
    for lemma in sorted(_TOKEN_DROP):
        rows.append(
            InventoryRow(
                lang="ar",
                level="*",
                lemma=lemma,
                english_lemma="",
                chinese_lemma="",
                upos="",
                action="drop",
                reason="classifier closed dialect/loan token set",
            )
        )
    # UPOS-conditional markers (apply uses classify_ar_lemma, not bare match alone).
    for lemma, upos, reason in (
        ("يعني#PART", "PART", "يعني discourse PART colloquial"),
        ("روح#VERB", "VERB", "colloquial imperative روح"),
        ("كون#SCONJ", "SCONJ", "Darija conditional كون"),
        ("لازم#AUX", "AUX", "لازم AUX Maghrebi"),
    ):
        rows.append(
            InventoryRow(
                lang="ar",
                level="*",
                lemma=lemma,
                english_lemma="",
                chinese_lemma="",
                upos=upos,
                action="drop",
                reason=reason,
            )
        )
    return rows


def apply_inventory_to_arabic_lists(root: Path, inventory: Sequence[InventoryRow]) -> int:
    """Drop rows that classify as drop; keep MSA exceptions. Returns drop count."""
    _ = inventory  # frozen inventory is audit trail; live classify is source of truth
    dropped = 0
    for level in LEVELS:
        for name in (f"{level}.csv", f"{level}.proposed.csv"):
            path = root / "arabic" / name
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = list(reader.fieldnames or [])
                rows = list(reader)
            lk = fields[0]
            out: list[dict[str, str]] = []
            for row in rows:
                lemma = _nfc(row.get(lk) or "")
                upos = (row.get("POS") or "").strip()
                en = _nfc(row.get("English_Lemma") or "")
                result = classify_ar_lemma(lemma, upos=upos, english=en)
                if result.action == "drop":
                    dropped += 1
                    continue
                if result.action == "policy":
                    zh = row.get("Chinese_Lemma") or ""
                    if "colloquial" not in en.lower() and "dialect" not in en.lower():
                        row["English_Lemma"] = f"{en} (colloquial)".strip()
                    if "口语" not in zh:
                        row["Chinese_Lemma"] = f"{zh}（口语）".strip()
                out.append(row)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(out)
    return dropped


# ---------------------------------------------------------------------------
# Pure sample scorer (plan criterion 4 + inventory join)
# ---------------------------------------------------------------------------


def score_sample_row(
    *,
    lang: str,
    level: str,
    lemma: str,
    english_lemma: str,
    chinese_lemma: str,
    upos: str,
    inventory_drops: Iterable[str] | None = None,
    inventory_policies: Iterable[str] | None = None,
) -> tuple[str, str]:
    """Return (verdict, notes). Deterministic; no bulk clean stamp."""
    lem = _nfc(lemma)
    en = _nfc(english_lemma)
    zh = _nfc(chinese_lemma)
    up = (upos or "").strip()
    drops = {strip_ar_diacritics(x) for x in (inventory_drops or [])}
    policies = set(inventory_policies or [])

    if not lem or not en or not zh:
        return "fix", "empty field"
    if lem != unicodedata.normalize("NFC", lem):
        return "fix", "lemma not NFC"
    if lang != "zh" and zh and re.search(r"[A-Za-z]", zh) and not re.search(
        r"[\u4e00-\u9fff]", zh
    ):
        return "fix", f"latin chinese_lemma {zh!r}"
    if lem in {"°", "º", "d'r"}:
        return "drop", "junk symbol/contraction"
    if lang == "sv" and lem == "los" and up == "ADJ":
        return "drop", "sv noise lemma"
    if lang == "de" and lem in {
        "meinten",
        "bräuchten",
        "nich",
        "wart",
        "Besonderes",
        "Heiliger",
        "Krachen",
        "Schwarzer",
        "ander",
    }:
        return "fix", "non-citation German form"
    if lang == "nl" and lem in {"uitdagingen", "honderden", "contracten"}:
        return "fix", "plural non-citation Dutch"
    if lang == "es" and lem == "émulo" and "emulator" in en.lower():
        return "fix", "false friend emulator"
    if lang == "nl" and lem == "zullen" and en.strip() == "would":
        return "fix", "zullen gloss"
    if lang == "ar" and lem == "قد" and "much" in en.lower():
        return "fix", "قد PART wrong gloss"
    if lang == "ar" and lem == "كمي" and "quantum" in en.lower():
        return "fix", "كمي false friend quantum"
    if lang == "ar" and lem == "إلا" and en.strip().lower() == "if":
        return "fix", "إلا means except/unless"
    if lang == "ar" and lem == "تكييف" and "qualification" in en.lower():
        return "fix", "تكييف wrong gloss"
    if lang == "ar" and lem == "قضى" and "errand" in en.lower():
        return "fix", "قضى overspecific gloss"
    if lang == "ar" and lem == "يعني" and up == "PART":
        return "drop", "يعني PART colloquial"
    # Classifier is source of truth (UPOS-conditional + MSA allow). Inventory
    # is the frozen closed lexicon audit trail; bare membership would false-drop
    # MSA exceptions such as روح NOUN.
    if lang == "ar":
        live = classify_ar_lemma(lem, upos=up, english=en)
        if live.action == "drop":
            return "drop", f"classifier:{live.reason}"
        if live.action == "policy" or lem in policies:
            return "keep", "policy:dialect-MSA-exception"
        bare = strip_ar_diacritics(lem)
        if bare in drops or lem in set(inventory_drops or []):
            # Closed-lexicon hit without UPOS context: only if not MSA allow.
            if live.reason != "msa_allow":
                return "drop", "inventory dialect residual still in list"
    return "keep", "clean"


__all__ = [
    "ClassifyResult",
    "InventoryRow",
    "apply_inventory_to_arabic_lists",
    "classify_ar_lemma",
    "closed_lexicon_inventory",
    "load_inventory",
    "scan_arabic_lists",
    "score_sample_row",
    "strip_ar_diacritics",
    "write_inventory",
]
