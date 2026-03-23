from __future__ import annotations

from collections import Counter
import math
import re
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

                                                  
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.config import settings
from app.core.rag import RagService
from app.core.retrieve import RetrievalHit

service = RagService()
web_dir = Path(__file__).parent / "web"

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


def _focus_quote(text: str, query: str, size: int = 420) -> str:
    exact_tokens = service.preprocessor.meaningful_exact_tokens(query)
    query_terms = {t for t in service.preprocessor.tokenize(query) if not service.preprocessor.is_morph_token(t)}
    sentences = [s.strip() for s in service.preprocessor.split_sentences(text) if s.strip()]
    if not sentences:
        return _clip(text, size=size)

    def sentence_relevance(sentence: str) -> tuple[float, int]:
        exact = _whole_word_match_count(sentence, exact_tokens)
        sentence_terms = set(service.preprocessor.tokenize(sentence))
        stem_overlap = len(query_terms & sentence_terms) if query_terms else 0
        score = float(exact * 3 + stem_overlap)
        if _is_sentence_fragment(sentence):
            score -= 0.8
        return score, exact

    ranked: list[tuple[int, float, int]] = []
    for idx, sentence in enumerate(sentences):
        score, exact = sentence_relevance(sentence)
        ranked.append((idx, score, exact))
    ranked.sort(key=lambda item: (item[1], item[2]), reverse=True)
    best_idx = ranked[0][0]

                                                                                     
    left = best_idx
    right = best_idx
    target_min_len = max(180, int(size * 0.62))
    max_len = max(size, target_min_len)

    def build_snippet(l: int, r: int) -> str:
        core = " ".join(sentences[l : r + 1]).strip()
        if l > 0:
            core = "... " + core
        if r < len(sentences) - 1:
            core = core + " ..."
        return core

                                                                               
    while len(build_snippet(left, right)) < target_min_len and (left > 0 or right < len(sentences) - 1):
        can_left = left > 0
        can_right = right < len(sentences) - 1
        if can_left and can_right:
            left_distance = best_idx - (left - 1)
            right_distance = (right + 1) - best_idx
            if left_distance <= right_distance:
                left -= 1
            else:
                right += 1
        elif can_left:
            left -= 1
        else:
            right += 1

                                                                                
    while len(build_snippet(left, right)) < target_min_len and (left > 0 or right < len(sentences) - 1):
        left_score = sentence_relevance(sentences[left - 1])[0] if left > 0 else -999.0
        right_score = sentence_relevance(sentences[right + 1])[0] if right < len(sentences) - 1 else -999.0
        if right_score > left_score and right < len(sentences) - 1:
            right += 1
        elif left > 0:
            left -= 1
        elif right < len(sentences) - 1:
            right += 1
        else:
            break

                                                 
    while len(build_snippet(left, right)) > max_len and (left < best_idx or right > best_idx):
        left_distance = best_idx - left
        right_distance = right - best_idx
        if right_distance >= left_distance and right > best_idx:
            right -= 1
        elif left < best_idx:
            left += 1
        else:
            break

                                                                  
    while left < best_idx and _is_sentence_fragment(sentences[left]):
        left += 1
    while right > best_idx and _is_sentence_fragment(sentences[right]):
        right -= 1

    snippet = build_snippet(left, right)

                                         
    for token in exact_tokens:
        pattern = re.compile(rf"(?<![A-Za-zА-Яа-яЁё0-9])({re.escape(token)})(?![A-Za-zА-Яа-яЁё0-9])", re.IGNORECASE)
        match = pattern.search(snippet)
        if match:
            s, e = match.span(1)
            snippet = snippet[:s] + "[[" + snippet[s:e] + "]]" + snippet[e:]
            break

    return snippet


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


@app.get("/health")
def health():
    books, chunks = service.stats()
    return jsonify({"status": "ok", "books": books, "chunks": chunks})


@app.get("/books")
def list_books():
    counts = _book_chunk_counts()
    books = [{"book": name, "chunks": counts[name]} for name in sorted(counts.keys())]
    return jsonify({"total": len(books), "books": books})


@app.post("/books/upload")
def upload_book():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"detail": "file is required"}), 400
    if not file.filename.lower().endswith(".txt"):
        return jsonify({"detail": "Only .txt files are supported"}), 400

    raw = file.read()
    content = None
    for enc in ("utf-8", "cp1251"):
        try:
            content = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if content is None:
        return jsonify({"detail": "File encoding must be UTF-8 or CP1251"}), 400
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

    snippets = [
        {
            "rank": start_idx + i + 1,
            "chunk_id": h.chunk.chunk_id,
            "book": h.chunk.book,
            "offset_start": h.chunk.offset_start,
            "offset_end": h.chunk.offset_end,
            "score": round(h.score, 4),
            "quote": _focus_quote(h.chunk.text, query),
        }
        for i, h in enumerate(hits)
    ]

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
            source_hits = _filter_relevant_hits(question, hits, strict=False)
        source_hits = _dedupe_hits(source_hits, top_k)
        result = service.answer_from_hits(question, source_hits)
    except Exception as exc:
        app.logger.exception("ask failed")
        return jsonify({"detail": f"ask failed: {type(exc).__name__}"}), 500

    source_hits = result.hits

    sources = [
        {
            "chunk_id": h.chunk.chunk_id,
            "book": h.chunk.book,
            "offset_start": h.chunk.offset_start,
            "offset_end": h.chunk.offset_end,
            "score": round(h.score, 4),
            "quote": _focus_quote(h.chunk.text, question),
        }
        for h in source_hits
    ]

    response = {"question": question_raw, "normalized_question": question, "answer": result.answer, "sources": sources}
    if allowed_books is not None:
        response["books_filter"] = sorted(allowed_books)
    if result.message:
        response["message"] = result.message
        response["suggestions"] = _question_suggestions(question, meaningful_terms)
    return jsonify(response)


if __name__ == "__main__":
    app.run(host=settings.host, port=settings.port, debug=True)
