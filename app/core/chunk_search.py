from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict

from app.config import settings
from app.core.preprocess import TextPreprocessor
from app.core.retrieve import RetrievalHit
from app.core.store import Chunk


class ChunkSearchEngine:
    """Fast lexical retriever with inverted index and hybrid BM25 scoring."""

    def __init__(self, preprocessor: TextPreprocessor) -> None:
        self.preprocessor = preprocessor
        self._lock = threading.RLock()
        self._chunks: list[Chunk] = []
        self._doc_term_freq: list[Counter[str]] = []
        self._doc_term_positions: list[dict[str, list[int]]] = []
        self._doc_exact_set: list[set[str]] = []
        self._doc_len: list[int] = []
        self._doc_freq: dict[str, int] = {}
        self._term_postings: dict[str, list[int]] = {}
        self._exact_postings: dict[str, list[int]] = {}
        self._avg_doc_len: float = 1.0
        self._ready = False

    def rebuild(self, chunks: list[Chunk]) -> None:
        if not chunks:
            with self._lock:
                self._chunks = []
                self._doc_term_freq = []
                self._doc_term_positions = []
                self._doc_exact_set = []
                self._doc_len = []
                self._doc_freq = {}
                self._term_postings = {}
                self._exact_postings = {}
                self._avg_doc_len = 1.0
                self._ready = False
            return

        doc_term_freq: list[Counter[str]] = []
        doc_term_positions: list[dict[str, list[int]]] = []
        doc_exact_set: list[set[str]] = []
        doc_len: list[int] = []
        term_postings: defaultdict[str, list[int]] = defaultdict(list)
        exact_postings: defaultdict[str, list[int]] = defaultdict(list)
        doc_freq: defaultdict[str, int] = defaultdict(int)

        for doc_idx, chunk in enumerate(chunks):
            doc_tokens = self.preprocessor.tokenize(chunk.text, include_synonyms=False)
            tf = Counter(doc_tokens)
            doc_term_freq.append(tf)
            doc_len.append(max(1, len(doc_tokens)))

            positions: defaultdict[str, list[int]] = defaultdict(list)
            for pos, token in enumerate(doc_tokens):
                positions[token].append(pos)
            doc_term_positions.append(dict(positions))

            exact_tokens = set(self.preprocessor.tokenize_exact(chunk.text))
            doc_exact_set.append(exact_tokens)

            for term in tf:
                term_postings[term].append(doc_idx)
                doc_freq[term] += 1
            for token in exact_tokens:
                exact_postings[token].append(doc_idx)

        avg_doc_len = max(1.0, sum(doc_len) / len(doc_len))
        with self._lock:
            self._chunks = chunks
            self._doc_term_freq = doc_term_freq
            self._doc_term_positions = doc_term_positions
            self._doc_exact_set = doc_exact_set
            self._doc_len = doc_len
            self._doc_freq = dict(doc_freq)
            self._term_postings = dict(term_postings)
            self._exact_postings = dict(exact_postings)
            self._avg_doc_len = avg_doc_len
            self._ready = True

    def _idf(self, term: str) -> float:
        n_docs = len(self._chunks)
        if n_docs <= 0:
            return 0.0
        df = self._doc_freq.get(term, 0)
        if df <= 0:
            return 0.0
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def _bm25_score(self, query_terms: list[str], doc_idx: int) -> tuple[float, int]:
        if not query_terms:
            return 0.0, 0
        k1 = 1.45
        b = 0.72
        tf = self._doc_term_freq[doc_idx]
        dl = self._doc_len[doc_idx]
        score = 0.0
        matched = 0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq <= 0:
                continue
            matched += 1
            idf = self._idf(term)
            if idf <= 0:
                continue
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * dl / self._avg_doc_len)
            score += idf * (numerator / denominator)
        return score, matched

    def _weighted_coverage(self, query_terms: list[str], doc_idx: int) -> float:
        if not query_terms:
            return 0.0
        tf = self._doc_term_freq[doc_idx]
        total = 0.0
        hit = 0.0
        for term in query_terms:
            weight = self._idf(term) or 0.05
            total += weight
            if tf.get(term, 0) > 0:
                hit += weight
        if total <= 0:
            return 0.0
        return hit / total

    def _proximity_score(self, query_terms: list[str], doc_idx: int) -> float:
        if not query_terms:
            return 0.0

        terms = sorted(set(query_terms))
        positions_map = self._doc_term_positions[doc_idx]
        terms = [term for term in terms if term in positions_map]
        if not terms:
            return 0.0
        if len(terms) == 1:
            return 1.0

        events: list[tuple[int, str]] = []
        for term in terms:
            for pos in positions_map[term]:
                events.append((pos, term))
        if not events:
            return 0.0
        events.sort(key=lambda item: item[0])

        need = len(terms)
        seen: defaultdict[str, int] = defaultdict(int)
        covered = 0
        left = 0
        best_span = math.inf
        best_coverage = 0.0

        for right, (_, term_right) in enumerate(events):
            if seen[term_right] == 0:
                covered += 1
            seen[term_right] += 1

            while left <= right:
                term_left = events[left][1]
                if seen[term_left] <= 1:
                    break
                seen[term_left] -= 1
                left += 1

            best_coverage = max(best_coverage, covered / need)
            if covered == need:
                span = float(events[right][0] - events[left][0] + 1)
                if span < best_span:
                    best_span = span

        if math.isinf(best_span):
            return 0.30 * best_coverage
        compactness = min(1.0, need / best_span)
        return 0.62 * best_coverage + 0.38 * compactness

    @staticmethod
    def _contains_query_phrase(text_norm: str, query_norm: str, *, whole_word: bool) -> bool:
        if not query_norm:
            return False
        if not whole_word:
            return query_norm in text_norm
        pattern = re.compile(
            rf"(?<![A-Za-zА-Яа-яЁё0-9\[\]]){re.escape(query_norm)}(?![A-Za-zА-Яа-яЁё0-9\[\]])",
            re.IGNORECASE,
        )
        return bool(pattern.search(text_norm))

    def _candidate_doc_ids(
        self,
        *,
        terms: set[str],
        exact_terms: set[str],
        allowed_books: set[str] | None,
    ) -> set[int]:
        candidates: set[int] = set()
        for term in terms:
            for doc_idx in self._term_postings.get(term, ()):
                candidates.add(doc_idx)
        for token in exact_terms:
            for doc_idx in self._exact_postings.get(token, ()):
                candidates.add(doc_idx)

        if allowed_books is not None:
            candidates = {idx for idx in candidates if self._chunks[idx].book in allowed_books}

        if not candidates and allowed_books is not None:
            return {idx for idx, chunk in enumerate(self._chunks) if chunk.book in allowed_books}
        return candidates

    def search(self, query: str, top_k: int, *, allowed_books: set[str] | None = None) -> list[RetrievalHit]:
        with self._lock:
            if not self._ready:
                return []

            top_k = max(1, min(top_k, settings.retrieval_top_k_max))
            core_terms = sorted(self.preprocessor.core_query_terms(query))
            expanded_terms = sorted(set(self.preprocessor.tokenize(query)))
            expanded_non_morph_terms = [t for t in expanded_terms if not self.preprocessor.is_morph_token(t)]
            exact_terms = sorted(set(self.preprocessor.meaningful_exact_tokens(query)))
            if not core_terms and not expanded_terms and not exact_terms:
                return []

            query_terms = set(core_terms) | set(expanded_terms)
            allowed = None if allowed_books is None else set(allowed_books)
            candidates = self._candidate_doc_ids(
                terms=query_terms,
                exact_terms=set(exact_terms),
                allowed_books=allowed,
            )
            if not candidates:
                return []

            core_min_terms = 0
            if len(core_terms) >= 3:
                core_min_terms = max(1, math.ceil(len(core_terms) * 0.5))
            query_norm = " ".join(query.lower().replace("ё", "е").split())
            short_single_exact = len(exact_terms) == 1 and len(exact_terms[0]) <= 4

            raw_scores: list[tuple[int, float]] = []
            for doc_idx in candidates:
                chunk = self._chunks[doc_idx]
                if allowed is not None and chunk.book not in allowed:
                    continue

                core_score, core_matches = self._bm25_score(core_terms, doc_idx)
                expanded_score, expanded_matches = self._bm25_score(expanded_terms, doc_idx)
                exact_matches = sum(1 for token in exact_terms if token in self._doc_exact_set[doc_idx])
                exact_ratio = (exact_matches / len(exact_terms)) if exact_terms else 0.0
                core_coverage = self._weighted_coverage(core_terms, doc_idx)
                expanded_coverage = self._weighted_coverage(expanded_non_morph_terms, doc_idx)
                proximity_terms = core_terms if len(core_terms) >= 2 else expanded_non_morph_terms
                proximity = self._proximity_score(proximity_terms, doc_idx)

                if core_matches == 0 and expanded_matches == 0 and exact_matches == 0:
                    continue
                if (
                    core_min_terms
                    and core_matches < core_min_terms
                    and exact_matches == 0
                    and core_coverage < 0.42
                    and proximity < 0.40
                ):
                    continue
                if short_single_exact and exact_matches == 0 and core_matches == 0 and expanded_coverage < 0.30:
                    continue

                if len(core_terms) <= 1:
                    lexical = 0.40 * core_score + 0.60 * expanded_score
                else:
                    lexical = 0.72 * core_score + 0.28 * expanded_score

                structure = 0.52 * core_coverage + 0.24 * expanded_coverage + 0.24 * proximity
                score = lexical * (0.42 + 0.58 * structure)
                score += 0.16 * exact_ratio
                score += 0.05 * min(1.0, expanded_matches / max(1, len(expanded_terms)))

                text_norm = chunk.text.lower().replace("ё", "е")
                if self._contains_query_phrase(text_norm, query_norm, whole_word=short_single_exact):
                    score += 0.14

                if exact_ratio == 0.0 and core_coverage < 0.20 and expanded_coverage < 0.20:
                    score *= 0.74

                if score > 0:
                    raw_scores.append((doc_idx, score))

            if not raw_scores:
                return []

            max_score = max(score for _, score in raw_scores)
            if max_score <= 0:
                return []

            ranked = sorted(raw_scores, key=lambda item: item[1], reverse=True)[:top_k]
            normalized_hits = [
                RetrievalHit(chunk=self._chunks[doc_idx], score=(score / max_score))
                for doc_idx, score in ranked
            ]

            threshold = settings.min_search_score
            if len(core_terms) <= 1:
                threshold = min(threshold, 0.03)
            elif len(core_terms) >= 4:
                threshold = max(threshold, 0.07)
            return [hit for hit in normalized_hits if hit.score >= threshold]
