from __future__ import annotations

import re
from collections import defaultdict

_RU_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она", "так",
    "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было",
    "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг",
    "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж",
    "вам", "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они", "тут", "где", "есть", "надо",
    "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем", "ним",
    "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "кажется", "сейчас", "были", "куда",
    "зачем", "всех", "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть", "после", "над",
    "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая", "много", "разве", "три", "эту",
    "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой",
    "им", "более", "всегда", "конечно", "всю", "между",
                                                                      
    "слово", "слова", "слов", "текст", "текста",
                                                             
    "найди", "найдите", "где", "говорится", "рассказывается", "упоминается", "описывается",
    "сказано", "написано", "известно", "расскажи", "покажи",
                                                                 
    "кто", "кого", "кому", "кем", "ком", "чей", "чья", "чье", "чьи",
    "каков", "какова", "каковы", "каково",
    "какой", "какая", "какое", "какие", "какого", "какой", "какую", "каком", "каким", "какими",
    "где", "куда", "откуда", "когда", "почему", "зачем", "сколько",
}

                                                                                     
_SYNONYM_SETS = [
    {"собака", "собачка", "собаку", "собаке", "собакой", "пес", "пёс", "пса", "псу", "псом", "песик", "пёсик", "щенок", "щенки"},
    {"кот", "котик", "кошка", "кота", "кошки", "котенка", "котенок", "котёнок"},
    {"лошадь", "конь", "лошадка", "коня", "конем", "конём"},
    {"лев", "льва", "льву", "львом", "льве"},
]

_SUFFIXES = (
    "иями", "ями", "ами", "ией", "иях", "иях", "ого", "ему", "ому", "ее", "ие", "ые", "ой", "ий", "ый", "ая",
    "ое", "ые", "ам", "ям", "ах", "ях", "ом", "ем", "ой", "ей", "ую", "юю", "а", "я", "ы", "и", "о", "е", "у",
)

_TOKEN_RE = re.compile(r"[a-zа-я0-9]+", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_MORPH_TOKEN_PREFIX = "m_"
_VOWELS = set("аеёиоуыэюя")
_QUERY_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"где\s+(?:говорится|рассказывается|упоминается|описывается)\s+(?:о|об|про)\s+|"
    r"(?:расскажи|покажи|найди|найдите)\s+(?:о|об|про)?\s*|"
    r"что\s+(?:сказано|написано|известно)\s+(?:о|об|про)\s+|"
    r"(?:есть|найдется|найдётся)\s+ли\s+(?:в\s+тексте\s+)?(?:о|об|про)\s+"
    r")",
    re.IGNORECASE,
)

_SPEECH_TERMS = {
    "говорил", "говорит", "сказал", "сказала", "сказали",
    "спросил", "спросила", "ответил", "ответила", "отвечал", "отвечала",
    "пишет", "писал", "писала", "написал", "написала", "произнес", "произнесла",
    "воскликнул", "воскликнула",
}


class TextPreprocessor:
    def __init__(self) -> None:
        self.synonym_map = self._build_synonym_map()

    def _build_synonym_map(self) -> dict[str, set[str]]:
        graph: dict[str, set[str]] = defaultdict(set)
        for group in _SYNONYM_SETS:
            normalized = {self._stem(self._normalize_token(t)) for t in group}
            for token in normalized:
                graph[token].update(normalized)
        return graph

    @staticmethod
    def _normalize_token(token: str) -> str:
        return token.lower().replace("ё", "е")

    @staticmethod
    def _stem(token: str) -> str:
                                                                                 
        if token == "лево":
            return "лево"

                                                                      
                                                             
        if token.startswith("лев") and len(token) >= 5 and token[3] in {"о", "ы", "у", "а", "е", "и"}:
            return "лево"

        if len(token) <= 2:
            return token
        for suffix in _SUFFIXES:
                                                                                       
                                                                                          
            if len(token) <= 5 and suffix in {"ий", "ый", "ой", "ее", "ие", "ые", "ая", "ое", "ую", "юю"}:
                continue

            min_stem_len = 3 if len(token) <= 5 else 4
            if token.endswith(suffix) and len(token) - len(suffix) >= min_stem_len:
                return token[: -len(suffix)]
        return token

    @staticmethod
    def _morph_token(stem: str) -> str | None:
                                                                               
                                                                           
                                                         
        if len(stem) < 3 or stem.startswith("лево"):
            return None
        consonants = "".join(ch for ch in stem if ch not in _VOWELS and ch not in {"ь", "ъ"})
        if len(consonants) < 2:
            return None
        return f"{_MORPH_TOKEN_PREFIX}{consonants}"

    @staticmethod
    def is_morph_token(token: str) -> bool:
        return token.startswith(_MORPH_TOKEN_PREFIX)

    def tokenize(self, text: str, *, include_synonyms: bool = True) -> list[str]:
        tokens = []
        for raw in _TOKEN_RE.findall(text.lower().replace("ё", "е")):
            if raw in _RU_STOPWORDS:
                continue
            stem = self._stem(raw)
            if stem in _RU_STOPWORDS or len(stem) < 2:
                continue
            tokens.append(stem)
            morph = self._morph_token(stem)
            if morph:
                tokens.append(morph)
            if include_synonyms and stem in self.synonym_map:
                tokens.extend(sorted(self.synonym_map[stem]))
        return tokens

    def core_query_terms(self, query: str) -> set[str]:
        return {t for t in self.tokenize(query, include_synonyms=False) if not self.is_morph_token(t)}

    def speech_query_terms(self, query: str) -> set[str]:
        return {t for t in self.core_query_terms(query) if t in _SPEECH_TERMS}

    def meaningful_query_terms(self, query: str) -> set[str]:
        core = self.core_query_terms(query)
        return {t for t in core if t not in _SPEECH_TERMS}

    def tokenize_exact(self, text: str) -> list[str]:
        tokens = []
        for raw in _TOKEN_RE.findall(text.lower().replace("ё", "е")):
            if raw in _RU_STOPWORDS or len(raw) < 2:
                continue
            tokens.append(raw)
        return tokens

    def meaningful_exact_tokens(self, query: str) -> list[str]:
        speech_terms = self.speech_query_terms(query)
        exact_tokens = self.tokenize_exact(query)
        if not speech_terms:
            return exact_tokens

        filtered = []
        for token in exact_tokens:
            stem = self._stem(self._normalize_token(token))
            if stem in speech_terms:
                continue
            filtered.append(token)
        return filtered

    def split_sentences(self, text: str) -> list[str]:
        sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
        return sentences if sentences else [text.strip()]

    def normalize_query(self, query: str) -> str:
        q = query.strip()
        q = _QUERY_PREFIX_RE.sub("", q)
        q = q.strip(" \t\n\r?!.,;:«»\"'()[]{}")
        return " ".join(q.split())
