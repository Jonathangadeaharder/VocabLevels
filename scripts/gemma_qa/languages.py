from __future__ import annotations

import unicodedata
from dataclasses import dataclass

LEVELS = ("A1", "A2", "B1", "B2", "C1")


@dataclass(frozen=True)
class LanguageProfile:
    directory: str
    code: str
    display_name: str
    endonym: str
    citation_rules: str

    @property
    def csv_header(self) -> str:
        return f"{self.directory.title()}_Lemma"


_PROFILES = (
    LanguageProfile(
        directory="english",
        code="en",
        display_name="English",
        endonym="English",
        citation_rules="Use dictionary citation lemmas; verbs must use the base form.",
    ),
    LanguageProfile(
        directory="german",
        code="de",
        display_name="German",
        endonym="Deutsch",
        citation_rules=(
            "Use German dictionary citation lemmas; common nouns are capitalized and "
            "verbs use the infinitive."
        ),
    ),
    LanguageProfile(
        directory="spanish",
        code="es",
        display_name="Spanish",
        endonym="Español",
        citation_rules="Use Spanish dictionary citation lemmas; verbs use the infinitive.",
    ),
    LanguageProfile(
        directory="arabic",
        code="ar",
        display_name="Arabic",
        endonym="العربية",
        citation_rules="Use the Arabic dictionary lemma written in Arabic script.",
    ),
    LanguageProfile(
        directory="french",
        code="fr",
        display_name="French",
        endonym="Français",
        citation_rules="Use French dictionary citation lemmas; verbs use the infinitive.",
    ),
    LanguageProfile(
        directory="swedish",
        code="sv",
        display_name="Swedish",
        endonym="Svenska",
        citation_rules="Use Swedish dictionary citation lemmas; verbs use the dictionary infinitive.",
    ),
    LanguageProfile(
        directory="chinese",
        code="zh",
        display_name="Chinese",
        endonym="中文",
        citation_rules="Use the uninflected written Chinese form as the lemma.",
    ),
    LanguageProfile(
        directory="dutch",
        code="nl",
        display_name="Dutch",
        endonym="Nederlands",
        citation_rules="Use Dutch dictionary citation lemmas; verbs use the dictionary infinitive.",
    ),
)

LANGUAGE_CODES = tuple(profile.code for profile in _PROFILES)
LANGUAGE_DIRECTORIES = tuple(profile.directory for profile in _PROFILES)
_BY_IDENTIFIER = {
    identifier: profile
    for profile in _PROFILES
    for identifier in (profile.code, profile.directory)
}


def get_language(identifier: str) -> LanguageProfile:
    try:
        return _BY_IDENTIFIER[identifier.casefold()]
    except KeyError as error:
        supported = ", ".join((*LANGUAGE_DIRECTORIES, *LANGUAGE_CODES))
        raise ValueError(
            f"unsupported language {identifier!r}; expected one of: {supported}"
        ) from error


def has_arabic_script(value: str) -> bool:
    return any(
        "\u0600" <= character <= "\u06ff"
        or "\u0750" <= character <= "\u077f"
        or "\u08a0" <= character <= "\u08ff"
        for character in value
    )


def has_han_script(value: str) -> bool:
    return any(
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
        for character in value
    )


def is_nfc(value: str) -> bool:
    return unicodedata.normalize("NFC", value) == value
