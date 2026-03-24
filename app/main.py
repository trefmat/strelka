from __future__ import annotations

from collections import Counter
from io import BytesIO
import html as html_lib
import math
import os
import re
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser

from flask import Flask, Response, jsonify, request, send_from_directory

                                                  
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.config import settings
from app.core.rag import RagService
from app.core.retrieve import RetrievalHit

service = RagService()
web_dir = Path(__file__).parent / "web"
preloaded_dir = Path(__file__).resolve().parent.parent / "examples" / "preloaded"

app = Flask(__name__, static_folder=str(web_dir), static_url_path="/web")

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
_NEGATIVE_CUE_RE = re.compile(
    r"\b(?:"
    r"не\s+(?:сказан[оаы]?|говорится|упоминается|описывается|найден[аоы]?)|"
    r"ничего\s+нет|"
    r"нет\s+упоминани[яй]"
    r")\b",
    re.IGNORECASE,
)
_STRONG_NEGATIVE_CUE_RE = re.compile(
    r"\bне\s+(?:сказан[оаы]?|говорится|упоминается|описывается|найден[аоы]?)\b",
    re.IGNORECASE,
)
_CAPITALIZED_TOKEN_RE = re.compile(r"\b[А-ЯЁ][а-яё]{2,}\b")
_NAME_STOPWORDS = {
    "это", "этот", "эта", "эти", "тот", "та", "те", "как", "что", "кто", "где", "когда", "почему",
    "зачем", "и", "но", "или", "а", "в", "во", "на", "по", "у", "к", "с", "со", "о", "об", "от", "до",
    "для", "из", "его", "ее", "её", "их", "она", "они", "он", "вы", "мы", "я", "ты", "все", "всё",
}
_WEAK_QUERY_TERMS = {
    "автор",
    "герой",
    "героиня",
    "геро",
    "персонаж",
    "персонажи",
    "сцена",
    "сцены",
    "сцен",
    "эпизод",
    "эпизоды",
    "подчеркивает",
    "описывает",
    "раскрывает",
    "показывает",
    "изображает",
    "рассказывает",
    "упоминает",
    "сообщает",
    "объясняет",
    "отмечает",
}
_WEAK_QUERY_PREFIXES = (
    "подчеркива",
    "подчерк",
    "описыва",
    "раскрыва",
    "показыва",
    "изобража",
    "рассказыва",
    "упомина",
    "сообща",
    "объясня",
    "отмеча",
)
_SUPPORTED_UPLOAD_EXTENSIONS = {".txt", ".fb2", ".epub"}
_MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))

app.config["MAX_CONTENT_LENGTH"] = _MAX_UPLOAD_BYTES


@app.errorhandler(413)
def payload_too_large(_err):
    max_mb = round(_MAX_UPLOAD_BYTES / (1024 * 1024), 1)
    return jsonify({"detail": f"File is too large. Max allowed size is {max_mb} MB"}), 413


def _clip(text: str, size: int = 360) -> str:
    return text if len(text) <= size else text[: size - 3] + "..."


def _normalize_quote(text: str) -> str:
    return " ".join(text.lower().split())


def _tokenize_norm(text: str) -> list[str]:
    return [t.lower().replace("ё", "е") for t in _TOKEN_RE.findall(text)]


def _range_overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0:
        return 0.0
    a_len = max(1, a_end - a_start)
    b_len = max(1, b_end - b_start)
    return inter / float(min(a_len, b_len))


def _token_jaccard(a: str, b: str) -> float:
    sa = set(_tokenize_norm(a))
    sb = set(_tokenize_norm(b))
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / float(union) if union else 0.0


def _difference_score(a: str, b: str) -> float:
                                                      
    return 1.0 - _token_jaccard(a, b)


def _negation_penalty(text: str) -> float:
    text_norm = " ".join(text.lower().replace("ё", "е").split())
    if _STRONG_NEGATIVE_CUE_RE.search(text_norm):
        return 0.78
    if _NEGATIVE_CUE_RE.search(text_norm):
        return 0.55
    return 0.0


def _whole_word_match_count(text: str, tokens: list[str]) -> int:
    if not tokens:
        return 0
    text_norm = text.lower().replace("ё", "е")
    total = 0
    for token in sorted(set(tokens), key=len, reverse=True):
        if len(token) < 2:
            continue
        pattern = re.compile(rf"(?<![A-Za-zА-Яа-яЁё0-9]){re.escape(token)}(?![A-Za-zА-Яа-яЁё0-9])")
        total += len(pattern.findall(text_norm))
    return total


def _is_sentence_fragment(sentence: str) -> bool:
    cleaned = sentence.strip().lstrip(" \t\n\r\"'«»()[]{}—-")
    if not cleaned:
        return True
    first = cleaned[0]
    if first in ",.;:!?":
        return True
    if first.islower():
        return True
    return False


def _focus_quote(chunk, query: str, size: int = 420, *, book_text: str | None = None) -> tuple[str, int, int]:
    text = chunk.text
    exact_tokens = service.preprocessor.meaningful_exact_tokens(query)
    query_terms = [t for t in service.preprocessor.tokenize(query) if not service.preprocessor.is_morph_token(t)]
    size = max(120, int(size))

    def find_anchor_span(haystack: str, tokens: list[str]) -> tuple[int, int] | None:
        for token in sorted(set(tokens), key=len, reverse=True):
            if len(token) < 2:
                continue
            pattern = re.compile(
                rf"(?<![A-Za-zА-Яа-яЁё0-9])({re.escape(token)})(?![A-Za-zА-Яа-яЁё0-9])",
                re.IGNORECASE,
            )
            match = pattern.search(haystack)
            if match:
                return match.span(1)
        return None

    def find_phrase_span(haystack: str, phrase: str) -> tuple[int, int] | None:
        probe = " ".join(phrase.lower().split())
        if len(probe) < 3:
            return None
        lower = haystack.lower()
        pos = lower.find(probe)
        if pos < 0:
            return None
        return pos, pos + len(probe)

    def nearest_literal_span(haystack: str, needle: str, expected_start: int, window: int = 5000) -> tuple[int, int] | None:
        if not needle:
            return None
        ws = max(0, expected_start - window)
        we = min(len(haystack), expected_start + window + len(needle))
        probe = haystack[ws:we]
        idx = probe.find(needle)
        if idx < 0:
            return None
        best_start = ws + idx
        best_dist = abs(best_start - expected_start)
        cursor = idx + 1
        while True:
            nxt = probe.find(needle, cursor)
            if nxt < 0:
                break
            abs_pos = ws + nxt
            dist = abs(abs_pos - expected_start)
            if dist < best_dist:
                best_dist = dist
                best_start = abs_pos
                if dist == 0:
                    break
            cursor = nxt + 1
        return best_start, best_start + len(needle)

    base_text = book_text
    if base_text is None:
        loaded_text, _ = _read_stored_book_text(chunk.book)
        base_text = loaded_text
    if not base_text:
        base_text = text

    chunk_start = max(0, min(int(chunk.offset_start), len(base_text)))
    chunk_end = max(chunk_start, min(int(chunk.offset_end), len(base_text)))
    if chunk_end <= chunk_start:
        chunk_start = 0
        chunk_end = len(base_text)

    chunk_text = base_text[chunk_start:chunk_end]
    if not chunk_text:
        chunk_text = text

    anchor_local = (
        find_phrase_span(chunk_text, query)
        or find_anchor_span(chunk_text, exact_tokens)
        or find_anchor_span(chunk_text, query_terms)
    )
    if anchor_local:
        expected_start = chunk_start + anchor_local[0]
        expected_end = chunk_start + anchor_local[1]
        anchor_piece = chunk_text[anchor_local[0]:anchor_local[1]]
        remap = nearest_literal_span(base_text, anchor_piece, expected_start)
        if remap:
            anchor_start, anchor_end = remap
        else:
            anchor_start, anchor_end = expected_start, expected_end
    else:
        local_in_original = (
            find_phrase_span(text, query)
            or find_anchor_span(text, exact_tokens)
            or find_anchor_span(text, query_terms)
        )
        if local_in_original:
            anchor_start = chunk_start + local_in_original[0]
            anchor_end = chunk_start + local_in_original[1]
        else:
            anchor_start = chunk_start + max(0, len(chunk_text) // 2)
            anchor_end = min(len(base_text), anchor_start + 1)

    center = (anchor_start + anchor_end) // 2
    half = size // 2
    start = max(0, center - half)
    end = min(len(base_text), start + size)
    start = max(0, end - size)

    core = base_text[start:end]
    mark_start = max(anchor_start, start)
    mark_end = min(anchor_end, end)
    if mark_end > mark_start:
        rel_s = mark_start - start
        rel_e = mark_end - start
        core = core[:rel_s] + "[[" + core[rel_s:rel_e] + "]]" + core[rel_e:]

    snippet = core.strip()
    if start > 0:
        snippet = "... " + snippet
    if end < len(base_text):
        snippet = snippet + " ..."
    if not snippet:
        snippet = _clip(text, size=size)
    return snippet, anchor_start, max(anchor_start + 1, anchor_end)


def _filter_relevant_hits(query: str, hits: list[RetrievalHit], *, strict: bool = False) -> list[RetrievalHit]:
    if not hits:
        return []

    raw_exact_tokens = service.preprocessor.meaningful_exact_tokens(query)
    core_query_terms = service.preprocessor.core_query_terms(query)
    meaningful_query_terms = service.preprocessor.meaningful_query_terms(query)
    speech_query_terms = service.preprocessor.speech_query_terms(query)
    expanded_query_terms = set(service.preprocessor.tokenize(query))
    query_norm = query.lower().replace("ё", "е").strip()
    anchor_terms = _effective_anchor_terms(meaningful_query_terms, core_query_terms)

    exact_tokens = []
    for token in raw_exact_tokens:
        stem = service.preprocessor._stem(service.preprocessor._normalize_token(token))
        if not anchor_terms or stem in anchor_terms:
            exact_tokens.append(token)
    if not exact_tokens:
        exact_tokens = raw_exact_tokens
    exact_variant_terms: list[set[str]] = []
    for token in exact_tokens:
        stem = service.preprocessor._stem(service.preprocessor._normalize_token(token))
        if not stem:
            continue
        variants = {stem}
        morph = None
        if len(stem) <= 4:
            morph = service.preprocessor._morph_token(stem)
        if morph:
            variants.add(morph)
        exact_variant_terms.append(variants)
    exact_query_set = set(exact_tokens)
    anchor_query_text = " ".join(sorted(anchor_terms))
    expanded_anchor_terms = set(service.preprocessor.tokenize(anchor_query_text)) if anchor_query_text else set()
    expanded_anchor_terms_non_morph = {t for t in expanded_anchor_terms if not service.preprocessor.is_morph_token(t)}

    scored: list[tuple[RetrievalHit, int, int, float, int]] = []
    for hit in hits:
        chunk_terms = set(service.preprocessor.tokenize(hit.chunk.text, include_synonyms=False))
        literal_exact_count = _whole_word_match_count(hit.chunk.text, exact_tokens)
        morph_exact_count = 0
        if exact_variant_terms:
            morph_exact_count = sum(1 for variants in exact_variant_terms if variants & chunk_terms)
        exact_count = max(literal_exact_count, morph_exact_count)
        morph_only_exact = literal_exact_count == 0 and morph_exact_count > 0
        core_overlap = len(core_query_terms & chunk_terms) if core_query_terms else 0
        anchor_overlap = len(anchor_terms & chunk_terms) if anchor_terms else 0
        speech_overlap = len(speech_query_terms & chunk_terms) if speech_query_terms else 0
        expanded_overlap = len(expanded_query_terms & chunk_terms) if expanded_query_terms else 0
        expanded_anchor_overlap = len(expanded_anchor_terms & chunk_terms) if expanded_anchor_terms else 0
        expanded_anchor_overlap_non_morph = (
            len(expanded_anchor_terms_non_morph & chunk_terms) if expanded_anchor_terms_non_morph else 0
        )

        sentences = service.preprocessor.split_sentences(hit.chunk.text)
        sentence_scores = service.retriever.sentence_scores(query, sentences)
        best_sentence_score = sentence_scores[0][1] if sentence_scores else 0.0

        if exact_count == 0 and anchor_overlap == 0 and best_sentence_score < 0.18:
            continue
        if len(anchor_terms) == 1 and len(exact_tokens) <= 1 and morph_only_exact and anchor_overlap == 0:
            has_non_fragment_signal = False
            sentence_score_map = {s: sc for s, sc in sentence_scores}
            for sentence in sentences:
                if _is_sentence_fragment(sentence):
                    continue
                sentence_terms = set(service.preprocessor.tokenize(sentence, include_synonyms=False))
                sentence_exact = _whole_word_match_count(sentence, exact_tokens)
                sentence_anchor_overlap = len(sentence_terms & anchor_terms) if anchor_terms else 0
                sentence_expanded_overlap_non_morph = (
                    len(sentence_terms & expanded_anchor_terms_non_morph) if expanded_anchor_terms_non_morph else 0
                )
                sentence_semantic_score = sentence_score_map.get(sentence, 0.0)
                if (
                    sentence_exact > 0
                    or sentence_anchor_overlap > 0
                    or sentence_expanded_overlap_non_morph > 0
                    or sentence_semantic_score >= 0.22
                ):
                    has_non_fragment_signal = True
                    break
            if not has_non_fragment_signal:
                                                                                   
                                                   
                continue
        if (
            len(anchor_terms) == 1
            and len(exact_tokens) <= 1
            and literal_exact_count == 0
            and anchor_overlap == 0
            and expanded_anchor_overlap_non_morph == 0
            and expanded_anchor_overlap > 0
            and best_sentence_score < 0.20
        ):
                                                                              
                                                          
            continue
        if len(anchor_terms) >= 2 and anchor_overlap == 0 and exact_count == 0 and best_sentence_score < 0.24:
            continue
        if strict and anchor_terms and anchor_overlap == 0 and expanded_anchor_overlap_non_morph == 0 and exact_count == 0:
            continue
        if strict and anchor_terms and len(anchor_terms) <= 2 and anchor_overlap == 0 and best_sentence_score < 0.30:
            continue

        exact_ratio = 0.0
        if exact_query_set:
            chunk_exact = set(service.preprocessor.tokenize_exact(hit.chunk.text))
            exact_ratio = len(chunk_exact & exact_query_set) / len(exact_query_set)
        if exact_variant_terms:
            morph_ratio = sum(1 for variants in exact_variant_terms if variants & chunk_terms) / len(exact_variant_terms)
            exact_ratio = max(exact_ratio, morph_ratio)

        core_coverage = (anchor_overlap / len(anchor_terms)) if anchor_terms else 0.0
        expanded_coverage = (
            (expanded_anchor_overlap_non_morph / len(expanded_anchor_terms_non_morph))
            if expanded_anchor_terms_non_morph
            else 0.0
        )
        coverage = max(core_coverage, expanded_coverage)
        speech_coverage = (speech_overlap / len(speech_query_terms)) if speech_query_terms else 0.0

        if strict and len(anchor_terms) >= 3 and coverage < 0.34 and exact_ratio < 0.30 and best_sentence_score < 0.48:
            continue
        if (not strict) and len(anchor_terms) >= 3 and coverage < 0.20 and exact_ratio == 0 and best_sentence_score < 0.40:
            continue

        phrase_bonus = 0.10 if query_norm and query_norm in hit.chunk.text.lower().replace("ё", "е") else 0.0
        negation_penalty = _negation_penalty(hit.chunk.text)
        if negation_penalty > 0 and len(anchor_terms) >= 2 and best_sentence_score < 0.78:
            continue
        rerank_score = (
            0.46 * hit.score
            + 0.29 * best_sentence_score
            + 0.17 * coverage
            + 0.08 * exact_ratio
            + phrase_bonus
        )
        if speech_query_terms:
            rerank_score += 0.05 * speech_coverage
            if strict and speech_overlap == 0 and best_sentence_score < 0.34:
                rerank_score *= 0.82

        if anchor_overlap == 0 and exact_count == 0:
            rerank_score *= 0.82
        if negation_penalty > 0:
            rerank_score *= (1.0 - negation_penalty)

        if strict and len(anchor_terms) >= 2 and rerank_score < 0.24:
            continue
        if (not strict) and len(anchor_terms) >= 2 and rerank_score < 0.14:
            continue

        scored.append(
            (
                RetrievalHit(chunk=hit.chunk, score=rerank_score),
                exact_count,
                anchor_overlap,
                best_sentence_score,
                expanded_anchor_overlap_non_morph,
            )
        )

    if not scored:
        return []

                                                                                         
    if any(exact > 0 for _, exact, _, _, _ in scored):
        if len(anchor_terms) == 1 and len(exact_tokens) == 1:
                                                                                        
                                                            
            semantic_floor = 0.36 if strict else 0.33
            scored = [item for item in scored if item[1] > 0 or item[4] > 0 or item[3] >= semantic_floor]
        else:
            scored = [item for item in scored if item[1] > 0 or item[3] >= 0.32]

    scored.sort(key=lambda item: (item[0].score, item[1], item[2], item[3], item[4]), reverse=True)

    best_score = scored[0][0].score
    if len(anchor_terms) >= 2:
        absolute_floor = 0.28 if strict else 0.16
        relative_floor = best_score * (0.72 if strict else 0.40)
    elif len(anchor_terms) == 1:
                                                                                           
        absolute_floor = 0.12 if strict else 0.06
        relative_floor = best_score * (0.34 if strict else 0.16)
    else:
        absolute_floor = 0.12
        relative_floor = best_score * (0.60 if strict else 0.54)
    floor = max(absolute_floor, relative_floor)

    filtered = [item[0] for item in scored if item[0].score >= floor]
    return filtered if filtered else [scored[0][0]]


def _dedupe_hits(hits: list[RetrievalHit], limit: int) -> list[RetrievalHit]:
    selected: list[RetrievalHit] = []
    selected_spans: set[tuple[str, int, int]] = set()
    seen_quotes: set[str] = set()

    def is_near_duplicate(candidate: RetrievalHit) -> bool:
        for selected_hit in selected:
            if selected_hit.chunk.book != candidate.chunk.book:
                continue
            overlap = _range_overlap_ratio(
                selected_hit.chunk.offset_start,
                selected_hit.chunk.offset_end,
                candidate.chunk.offset_start,
                candidate.chunk.offset_end,
            )
            sim = _token_jaccard(selected_hit.chunk.text, candidate.chunk.text)
            if overlap >= 0.20:
                return True
            if sim >= 0.84:
                return True
            if overlap >= 0.10 and sim >= 0.70:
                return True
            if _difference_score(selected_hit.chunk.text, candidate.chunk.text) <= 0.20:
                return True
        return False

    for hit in hits:
        span_key = (hit.chunk.book, hit.chunk.offset_start, hit.chunk.offset_end)
        if span_key in selected_spans:
            continue
        quote_key = _normalize_quote(_clip(hit.chunk.text, size=220))
        if quote_key in seen_quotes:
            continue
        if is_near_duplicate(hit):
            continue

        selected.append(hit)
        selected_spans.add(span_key)
        seen_quotes.add(quote_key)

        if len(selected) >= limit:
            break

    return selected


def _extend_with_fallback_hits(
    primary_hits: list[RetrievalHit],
    fallback_hits: list[RetrievalHit],
    *,
    limit: int,
) -> list[RetrievalHit]:
    if len(primary_hits) >= limit:
        return primary_hits[:limit]

    selected = list(primary_hits)
    seen_spans = {(h.chunk.book, h.chunk.offset_start, h.chunk.offset_end) for h in selected}

    for hit in fallback_hits:
        if len(selected) >= limit:
            break
        span_key = (hit.chunk.book, hit.chunk.offset_start, hit.chunk.offset_end)
        if span_key in seen_spans:
            continue
        selected.append(hit)
        seen_spans.add(span_key)

    return selected


def _lexical_fallback_hits(query: str, top_k: int, *, allowed_books: set[str] | None = None) -> list[RetrievalHit]:
    exact_tokens = service.preprocessor.meaningful_exact_tokens(query)
    core_terms = service.preprocessor.core_query_terms(query)
    expanded_terms = set(service.preprocessor.tokenize(query))
    if not exact_tokens and not core_terms and not expanded_terms:
        return []

    query_norm = query.lower().replace("ё", "е").strip()
    candidates: list[tuple[float, RetrievalHit]] = []

    for chunk in service.store.all_chunks():
        if allowed_books is not None and chunk.book not in allowed_books:
            continue
        exact_count = _whole_word_match_count(chunk.text, exact_tokens)
        chunk_terms = set(service.preprocessor.tokenize(chunk.text, include_synonyms=False))
        core_overlap = len(core_terms & chunk_terms) if core_terms else 0
        expanded_overlap = len(expanded_terms & chunk_terms) if expanded_terms else 0
        if exact_count == 0 and core_overlap == 0 and expanded_overlap == 0:
            continue

        phrase_bonus = 0.20 if query_norm and query_norm in chunk.text.lower().replace("ё", "е") else 0.0
        raw_score = exact_count * 1.00 + core_overlap * 0.45 + expanded_overlap * 0.30 + phrase_bonus
        candidates.append((raw_score, RetrievalHit(chunk=chunk, score=raw_score)))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    top = candidates[: max(1, top_k)]
    max_raw = top[0][0] if top else 1.0
    if max_raw <= 0:
        return [hit for _, hit in top]
    return [RetrievalHit(chunk=hit.chunk, score=(raw / max_raw)) for raw, hit in top]


def _exact_match_hits(query: str, top_k: int, *, allowed_books: set[str] | None = None) -> list[RetrievalHit]:
    exact_tokens = service.preprocessor.meaningful_exact_tokens(query)
    if not exact_tokens:
        return []

    query_norm = query.lower().replace("ё", "е").strip()
    candidates: list[tuple[float, RetrievalHit]] = []

    for chunk in service.store.all_chunks():
        if allowed_books is not None and chunk.book not in allowed_books:
            continue
        exact_count = _whole_word_match_count(chunk.text, exact_tokens)
        if exact_count <= 0:
            continue

        phrase_bonus = 0.15 if query_norm and query_norm in chunk.text.lower().replace("ё", "е") else 0.0
        raw_score = exact_count + phrase_bonus
        candidates.append((raw_score, RetrievalHit(chunk=chunk, score=raw_score)))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    top = candidates[: max(1, top_k)]
    max_raw = top[0][0] if top else 1.0
    if max_raw <= 0:
        return [hit for _, hit in top]
    return [RetrievalHit(chunk=hit.chunk, score=(raw / max_raw)) for raw, hit in top]


def _is_weak_query_term(term: str) -> bool:
    return term in _WEAK_QUERY_TERMS or any(term.startswith(prefix) for prefix in _WEAK_QUERY_PREFIXES)


def _effective_anchor_terms(meaningful_terms: set[str], core_terms: set[str]) -> set[str]:
    anchor_terms = meaningful_terms or core_terms
    if not anchor_terms:
        return set()
    strong_terms = {term for term in anchor_terms if not _is_weak_query_term(term)}
    return strong_terms if strong_terms else anchor_terms


def _book_chunk_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    for chunk in service.store.all_chunks():
        counts[chunk.book] += 1
    return counts


def _parse_books_filter(payload: dict) -> tuple[set[str] | None, str | None]:
    raw_books = payload.get("books")
    if raw_books is None:
        return None, None
    if not isinstance(raw_books, list):
        return None, "books must be an array of strings"

    normalized: list[str] = []
    for item in raw_books:
        if not isinstance(item, str):
            return None, "books must be an array of strings"
        name = Path(item).name.strip()
        if name:
            normalized.append(name)

    selected = set(normalized)
    if not selected:
        return set(), None

    available = set(_book_chunk_counts().keys())
    unknown = sorted(selected - available)
    if unknown:
        return None, f"unknown books: {', '.join(unknown)}"
    return selected, None


def _hit_has_direct_support(query: str, hit: RetrievalHit) -> bool:
    exact_tokens = service.preprocessor.meaningful_exact_tokens(query)
    if _whole_word_match_count(hit.chunk.text, exact_tokens) > 0:
        return True

    core_terms = service.preprocessor.core_query_terms(query)
    if core_terms:
        chunk_terms = set(service.preprocessor.tokenize(hit.chunk.text, include_synonyms=False))
        overlap = len(core_terms & chunk_terms)
        min_overlap = 1 if len(core_terms) <= 2 else 2
        if overlap >= min_overlap:
            return True

    return False


def _parse_quote_size(payload: dict, *, default: int = 420, min_size: int = 120, max_size: int = 2400) -> tuple[int, str | None]:
    raw = payload.get("quote_size", default)
    try:
        size = int(raw)
    except (TypeError, ValueError):
        return default, "quote_size must be integer"
    if size < min_size or size > max_size:
        return default, f"quote_size must be between {min_size} and {max_size}"
    return size, None


class _HtmlTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in {"p", "div", "li", "section", "article"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if data and data.strip():
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def _normalize_extracted_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    non_empty = [line for line in lines if line]
    return "\n".join(non_empty).strip()


def _decode_text_bytes(raw: bytes) -> str | None:
    for enc in ("utf-8-sig", "utf-8", "cp1251", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def _extract_fb2_text(raw: bytes) -> str | None:
    decoded = _decode_text_bytes(raw)
    if decoded is None:
        return None
    try:
        root = ET.fromstring(decoded)
    except ET.ParseError:
        return None
    parts: list[str] = []
    for elem in root.iter():
        if elem.text and elem.text.strip():
            parts.append(elem.text.strip())
    if not parts:
        return None
    return _normalize_extracted_text("\n".join(parts))


def _extract_epub_text(raw: bytes) -> str | None:
    try:
        archive = zipfile.ZipFile(BytesIO(raw))
    except zipfile.BadZipFile:
        return None

    with archive:
        names = set(archive.namelist())
        opf_path = None
        if "META-INF/container.xml" in names:
            try:
                container_xml = archive.read("META-INF/container.xml")
                container_root = ET.fromstring(container_xml)
                for elem in container_root.iter():
                    if elem.tag.endswith("rootfile"):
                        candidate = elem.attrib.get("full-path")
                        if candidate:
                            opf_path = candidate
                            break
            except Exception:
                opf_path = None

        html_paths: list[str] = []
        if opf_path and opf_path in names:
            try:
                opf_xml = archive.read(opf_path)
                opf_root = ET.fromstring(opf_xml)
                base_dir = str(Path(opf_path).parent).replace("\\", "/")
                manifest_by_id: dict[str, str] = {}
                for elem in opf_root.iter():
                    if elem.tag.endswith("item"):
                        item_id = elem.attrib.get("id")
                        href = elem.attrib.get("href")
                        media = (elem.attrib.get("media-type") or "").lower()
                        if not item_id or not href:
                            continue
                        if "html" in media or href.lower().endswith((".xhtml", ".html", ".htm")):
                            joined = f"{base_dir}/{href}" if base_dir and base_dir != "." else href
                            manifest_by_id[item_id] = joined
                for elem in opf_root.iter():
                    if elem.tag.endswith("itemref"):
                        item_idref = elem.attrib.get("idref")
                        if item_idref and item_idref in manifest_by_id:
                            html_paths.append(manifest_by_id[item_idref])
            except Exception:
                html_paths = []

        if not html_paths:
            html_paths = sorted(
                name for name in names
                if name.lower().endswith((".xhtml", ".html", ".htm")) and not name.lower().startswith("meta-inf/")
            )

        extracted_blocks: list[str] = []
        for path in html_paths:
            if path not in names:
                continue
            try:
                content = archive.read(path)
            except KeyError:
                continue
            decoded = _decode_text_bytes(content)
            if decoded is None:
                continue
            parser = _HtmlTextExtractor()
            try:
                parser.feed(decoded)
                parser.close()
            except Exception:
                continue
            text = _normalize_extracted_text(html_lib.unescape(parser.text()))
            if text:
                extracted_blocks.append(text)

        if not extracted_blocks:
            return None
        return _normalize_extracted_text("\n\n".join(extracted_blocks))


def _extract_upload_text(filename: str, raw: bytes) -> tuple[str | None, str | None]:
    ext = Path(filename).suffix.lower()
    if ext == ".txt":
        decoded = _decode_text_bytes(raw)
        if decoded is None:
            return None, "File encoding must be UTF-8, UTF-16 or CP1251"
        return decoded, None
    if ext == ".fb2":
        text = _extract_fb2_text(raw)
        if not text:
            return None, "Failed to parse FB2 file"
        return text, None
    if ext == ".epub":
        text = _extract_epub_text(raw)
        if not text:
            return None, "Failed to parse EPUB file"
        return text, None
    return None, "Only .txt, .fb2 and .epub files are supported"


def _preloaded_books() -> list[dict]:
    if not preloaded_dir.exists():
        return []
    items: list[dict] = []
    for path in sorted(preloaded_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _SUPPORTED_UPLOAD_EXTENSIONS:
            continue
        items.append({"book": path.name, "size_bytes": path.stat().st_size})
    return items


def _read_stored_book_text(book: str) -> tuple[str | None, str | None]:
    name = Path(book).name.strip()
    if not name:
        return None, "book is required"
    path = service.store.books_dir / name
    if not path.exists() or not path.is_file():
        return None, f"book not found: {name}"
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, f"failed to read book: {name} ({exc})"
    decoded = _decode_text_bytes(raw)
    if decoded is None:
        return None, f"failed to decode book: {name}"
    normalized = decoded.replace("\r\n", "\n").replace("\r", "\n")
    return normalized, None


def _book_subject_hints(limit: int = 3, scan_chunks: int = 500) -> list[str]:
    counts: Counter[str] = Counter()
    display_tokens: dict[str, str] = {}

    for idx, chunk in enumerate(service.store.all_chunks()):
        if idx >= scan_chunks:
            break
        for token in _CAPITALIZED_TOKEN_RE.findall(chunk.text):
            key = token.lower().replace("ё", "е")
            if key in _NAME_STOPWORDS:
                continue
            counts[key] += 1
            display_tokens.setdefault(key, token)

    if not counts:
        return []

    hints: list[str] = []
    for key, _ in counts.most_common(limit * 3):
        token = display_tokens.get(key, key.capitalize())
        if token in hints:
            continue
        hints.append(token)
        if len(hints) >= limit:
            break
    return hints


def _question_suggestions(question: str, meaningful_terms: set[str] | None = None, *, limit: int = 3) -> list[str]:
    terms = sorted(t for t in (meaningful_terms or set()) if len(t) >= 3)
    if not terms:
        terms = sorted(t for t in service.preprocessor.meaningful_query_terms(question) if len(t) >= 3)

    suggestions: list[str] = []
    if len(terms) >= 2:
        first, second = terms[0], terms[1]
        suggestions.extend(
            [
                f"Как связаны «{first}» и «{second}» в одном эпизоде?",
                f"Что меняется в сюжете после сцены с «{first}» и «{second}»?",
                f"Как автор противопоставляет «{first}» и «{second}»?",
            ]
        )
    elif len(terms) == 1:
        term = terms[0]
        suggestions.extend(
            [
                f"В каком эпизоде подробно говорится о «{term}»?",
                f"Как «{term}» влияет на ход событий?",
                f"Что автор подчеркивает, когда описывает «{term}»?",
            ]
        )
    else:
        hints = _book_subject_hints(limit=3)
        if hints:
            lead = hints[0]
            suggestions.extend(
                [
                    f"В каком эпизоде «{lead}» принимает ключевое решение?",
                    f"Как меняется образ «{lead}» по ходу книги?",
                    f"Что автор сообщает о «{lead}» в разных сценах?",
                ]
            )
            if len(hints) > 1:
                suggestions.append(f"Как связаны «{hints[0]}» и «{hints[1]}» в сюжете?")
        else:
            suggestions.extend(
                [
                    "Что происходит с конкретным персонажем в выбранной сцене?",
                    "Почему герой совершает это действие и к чему это приводит?",
                    "Как автор описывает ключевое событие и его последствия?",
                ]
            )

    unique: list[str] = []
    seen: set[str] = set()
    for suggestion in suggestions:
        key = suggestion.lower().replace("ё", "е")
        if key in seen:
            continue
        seen.add(key)
        unique.append(suggestion)
        if len(unique) >= limit:
            break
    return unique


@app.get("/")
def index():
    return send_from_directory(str(web_dir), "index.html")


@app.get("/reader")
def reader():
    return send_from_directory(str(web_dir), "reader.html")


@app.get("/health")
def health():
    books, chunks = service.stats()
    return jsonify({"status": "ok", "books": books, "chunks": chunks})


@app.get("/books")
def list_books():
    counts = _book_chunk_counts()
    books = [{"book": name, "chunks": counts[name]} for name in sorted(counts.keys())]
    return jsonify({"total": len(books), "books": books})


@app.get("/books/content")
def book_content():
    book = str(request.args.get("book", "")).strip()
    text, error = _read_stored_book_text(book)
    if error:
        code = 404 if "not found" in error else 400
        return jsonify({"detail": error}), code
    return Response(text, mimetype="text/plain; charset=utf-8")


@app.get("/books/preloaded")
def list_preloaded_books():
    items = _preloaded_books()
    return jsonify({"total": len(items), "books": items})


@app.post("/books/load_preloaded")
def load_preloaded_book():
    payload = request.get_json(silent=True) or {}
    book = Path(str(payload.get("book", "")).strip()).name
    if not book:
        return jsonify({"detail": "book is required"}), 400
    if Path(book).suffix.lower() not in _SUPPORTED_UPLOAD_EXTENSIONS:
        return jsonify({"detail": "Only .txt, .fb2 and .epub files are supported"}), 400

    path = preloaded_dir / book
    if not path.exists() or not path.is_file():
        return jsonify({"detail": f"preloaded book not found: {book}"}), 404

    raw = path.read_bytes()
    if len(raw) > _MAX_UPLOAD_BYTES:
        max_mb = round(_MAX_UPLOAD_BYTES / (1024 * 1024), 1)
        return jsonify({"detail": f"File is too large. Max allowed size is {max_mb} MB"}), 413
    content, error = _extract_upload_text(book, raw)
    if error:
        return jsonify({"detail": error}), 400
    if not content.strip():
        return jsonify({"detail": "Preloaded book is empty"}), 400

    added = service.upload_book(book, content)
    return jsonify({"book": book, "chunks_added": added, "message": "Preloaded book uploaded and indexed"})


@app.post("/books/upload")
def upload_book():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"detail": "file is required"}), 400
    if Path(file.filename).suffix.lower() not in _SUPPORTED_UPLOAD_EXTENSIONS:
        return jsonify({"detail": "Only .txt, .fb2 and .epub files are supported"}), 400

    raw = file.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        max_mb = round(_MAX_UPLOAD_BYTES / (1024 * 1024), 1)
        return jsonify({"detail": f"File is too large. Max allowed size is {max_mb} MB"}), 413
    content, error = _extract_upload_text(file.filename, raw)
    if error:
        return jsonify({"detail": error}), 400
    if not content.strip():
        return jsonify({"detail": "Book is empty"}), 400

    added = service.upload_book(file.filename, content)
    return jsonify({"book": file.filename, "chunks_added": added, "message": "Book uploaded and indexed"})


@app.post("/search/snippets")
def search_snippets():
    payload = request.get_json(silent=True) or {}
    query_raw = str(payload.get("query", "")).strip()
    query = service.preprocessor.normalize_query(query_raw)
    if not query:
        return jsonify({"detail": "query is required"}), 400
    allowed_books, filter_error = _parse_books_filter(payload)
    if filter_error:
        return jsonify({"detail": filter_error}), 400
    quote_size, quote_error = _parse_quote_size(payload)
    if quote_error:
        return jsonify({"detail": quote_error}), 400

    try:
        page = int(payload.get("page", 1))
    except (TypeError, ValueError):
        return jsonify({"detail": "page must be integer"}), 400
    try:
        page_size = int(payload.get("page_size", payload.get("top_k", settings.search_page_size_default)))
    except (TypeError, ValueError):
        return jsonify({"detail": "page_size must be integer"}), 400

    page = max(1, page)
    page_size = max(1, min(page_size, settings.search_page_size_max))

    all_chunks = service.store.all_chunks()
    if allowed_books is None:
        scoped_chunk_count = len(all_chunks)
    else:
        scoped_chunk_count = sum(1 for chunk in all_chunks if chunk.book in allowed_books)
    all_chunk_count = max(1, scoped_chunk_count)

    try:
        semantic_hits = service.search_snippets(
            query,
            top_k=settings.retrieval_top_k_max,
            allowed_books=allowed_books,
        )
        fallback_hits = _lexical_fallback_hits(query, top_k=all_chunk_count, allowed_books=allowed_books)
        if semantic_hits:
            mixed_hits = _extend_with_fallback_hits(
                semantic_hits,
                fallback_hits,
                limit=max(len(semantic_hits), len(fallback_hits)),
            )
        else:
            mixed_hits = fallback_hits

        exact_hits = _exact_match_hits(query, top_k=all_chunk_count, allowed_books=allowed_books)
        if exact_hits:
            raw_hits = _extend_with_fallback_hits(
                exact_hits,
                mixed_hits,
                limit=max(len(exact_hits), len(mixed_hits)),
            )
            match_mode = "exact_priority"
        else:
            raw_hits = mixed_hits
            match_mode = "mixed"
    except Exception as exc:
        app.logger.exception("search_snippets failed")
        return jsonify({"detail": f"search failed: {type(exc).__name__}"}), 500

    filtered_hits = _filter_relevant_hits(query, raw_hits)
    all_hits = _dedupe_hits(filtered_hits, max(1, len(filtered_hits))) if filtered_hits else []

    total = len(all_hits)
    total_pages = max(1, math.ceil(total / page_size))
    page = min(page, total_pages)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    hits = all_hits[start_idx:end_idx]

    book_text_cache: dict[str, str] = {}
    snippets = []
    for i, h in enumerate(hits):
        if h.chunk.book not in book_text_cache:
            text_cached, _ = _read_stored_book_text(h.chunk.book)
            book_text_cache[h.chunk.book] = text_cached or ""
        quote, focus_start, focus_end = _focus_quote(
            h.chunk,
            query,
            size=quote_size,
            book_text=book_text_cache.get(h.chunk.book),
        )
        snippets.append(
            {
                "rank": start_idx + i + 1,
                "chunk_id": h.chunk.chunk_id,
                "book": h.chunk.book,
                "offset_start": h.chunk.offset_start,
                "offset_end": h.chunk.offset_end,
                "focus_start": focus_start,
                "focus_end": focus_end,
                "score": round(h.score, 4),
                "quote": quote,
            }
        )

    result = {
        "query": query_raw,
        "normalized_query": query,
        "snippets": snippets,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "has_prev": page > 1,
        "has_next": page < total_pages,
        "match_mode": match_mode,
        "quote_size": quote_size,
    }
    if allowed_books is not None:
        result["books_filter"] = sorted(allowed_books)
    if not snippets:
        result["message"] = "not_found"
    return jsonify(result)


@app.post("/ask")
def ask():
    payload = request.get_json(silent=True) or {}
    question_raw = str(payload.get("question", "")).strip()
    question = service.preprocessor.normalize_query(question_raw)
    if not question:
        return jsonify({"detail": "question is required"}), 400
    allowed_books, filter_error = _parse_books_filter(payload)
    if filter_error:
        return jsonify({"detail": filter_error}), 400
    quote_size, quote_error = _parse_quote_size(payload)
    if quote_error:
        return jsonify({"detail": quote_error}), 400

    core_terms = service.preprocessor.core_query_terms(question)
    meaningful_terms = service.preprocessor.meaningful_query_terms(question)
    if not core_terms or not meaningful_terms:
        response = {
            "question": question_raw,
            "normalized_question": question,
            "answer": "Вопрос слишком общий. Уточните объект поиска: персонажа, событие или тему.",
            "suggestions": _question_suggestions(question, meaningful_terms),
            "sources": [],
            "message": "clarify_needed",
        }
        if allowed_books is not None:
            response["books_filter"] = sorted(allowed_books)
        return jsonify(response)
    try:
        top_k = int(payload.get("top_k", settings.retrieval_top_k_default))
    except (TypeError, ValueError):
        return jsonify({"detail": "top_k must be integer"}), 400

    top_k = max(1, min(top_k, settings.retrieval_top_k_max))
    retrieve_k = max(top_k, min(settings.retrieval_top_k_max, top_k * 4))
    try:
        hits = service.search_snippets(question, top_k=retrieve_k, allowed_books=allowed_books)
        if not hits:
            hits = _lexical_fallback_hits(question, top_k=retrieve_k, allowed_books=allowed_books)
        source_hits = _filter_relevant_hits(question, hits, strict=True)
        if not source_hits:
            loose_hits = _filter_relevant_hits(question, hits, strict=False)
            source_hits = [h for h in loose_hits if _hit_has_direct_support(question, h)]
        source_hits = _dedupe_hits(source_hits, top_k)
        result = service.answer_from_hits(question, source_hits)
    except Exception as exc:
        app.logger.exception("ask failed")
        return jsonify({"detail": f"ask failed: {type(exc).__name__}"}), 500

    source_hits = result.hits

    source_cache: dict[str, str] = {}
    sources = []
    for h in source_hits:
        if h.chunk.book not in source_cache:
            text_cached, _ = _read_stored_book_text(h.chunk.book)
            source_cache[h.chunk.book] = text_cached or ""
        quote, focus_start, focus_end = _focus_quote(
            h.chunk,
            question,
            size=quote_size,
            book_text=source_cache.get(h.chunk.book),
        )
        sources.append(
            {
                "chunk_id": h.chunk.chunk_id,
                "book": h.chunk.book,
                "offset_start": h.chunk.offset_start,
                "offset_end": h.chunk.offset_end,
                "focus_start": focus_start,
                "focus_end": focus_end,
                "score": round(h.score, 4),
                "quote": quote,
            }
        )

    response = {
        "question": question_raw,
        "normalized_question": question,
        "answer": result.answer,
        "sources": sources,
        "quote_size": quote_size,
    }
    if allowed_books is not None:
        response["books_filter"] = sorted(allowed_books)
    if result.message:
        response["message"] = result.message
        response["suggestions"] = _question_suggestions(question, meaningful_terms)
    return jsonify(response)


if __name__ == "__main__":
    app.run(host=settings.host, port=settings.port, debug=True)
