from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass

from app.config import settings
from app.core.preprocess import TextPreprocessor
from app.core.store import Chunk


@dataclass(slots=True)
class RetrievalHit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, preprocessor: TextPreprocessor) -> None:
        self.preprocessor = preprocessor
        self._lock = threading.RLock()
        self._chunks: list[Chunk] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_term_freq: list[Counter[str]] = []
        self._doc_term_positions: list[dict[str, list[int]]] = []
        self._doc_exact_set: list[set[str]] = []
        self._doc_len: list[int] = []
        self._doc_freq: dict[str, int] = {}
        self._avg_doc_len: float = 1.0
        self._fitted = False

    def rebuild(self, chunks: list[Chunk]) -> None:
                                                                                
        if not chunks:
            with self._lock:
                self._chunks = []
                self._fitted = False
                self._doc_tokens = []
                self._doc_term_freq = []
                self._doc_term_positions = []
                self._doc_exact_set = []
                self._doc_len = []
                self._doc_freq = {}
                self._avg_doc_len = 1.0
            return

        doc_tokens = [self.preprocessor.tokenize(c.text, include_synonyms=False) for c in chunks]
        doc_term_freq = [Counter(tokens) for tokens in doc_tokens]
        doc_term_positions = []
        for tokens in doc_tokens:
            positions: defaultdict[str, list[int]] = defaultdict(list)
            for idx, token in enumerate(tokens):
                positions[token].append(idx)
            doc_term_positions.append(dict(positions))
        doc_exact_set = [set(self.preprocessor.tokenize_exact(c.text)) for c in chunks]
        doc_len = [max(1, len(tokens)) for tokens in doc_tokens]
        avg_doc_len = max(1.0, sum(doc_len) / len(doc_len))

        df: defaultdict[str, int] = defaultdict(int)
        for tokens in doc_tokens:
            for token in set(tokens):
                df[token] += 1
        doc_freq = dict(df)

        with self._lock:
            self._chunks = chunks
            self._doc_tokens = doc_tokens
            self._doc_term_freq = doc_term_freq
            self._doc_term_positions = doc_term_positions
            self._doc_exact_set = doc_exact_set
            self._doc_len = doc_len
            self._doc_freq = doc_freq
            self._avg_doc_len = avg_doc_len
            self._fitted = True

    def _idf(self, term: str) -> float:
        n_docs = len(self._chunks)
        df = self._doc_freq.get(term, 0)
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def _bm25_score(self, query_terms: list[str], doc_idx: int) -> tuple[float, int]:
        k1 = 1.5
        b = 0.75
        tf = self._doc_term_freq[doc_idx]
        dl = self._doc_len[doc_idx]
        score = 0.0
        matched_terms = 0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            matched_terms += 1
            idf = self._idf(term)
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * dl / self._avg_doc_len)
            score += idf * (numerator / denominator)
        return score, matched_terms

    def _weighted_coverage(self, query_terms: list[str], doc_idx: int) -> float:
        if not query_terms:
            return 0.0
        tf = self._doc_term_freq[doc_idx]
        weighted_total = 0.0
        weighted_hit = 0.0
        for term in query_terms:
            weight = self._idf(term)
            weighted_total += weight
            if tf.get(term, 0) > 0:
                weighted_hit += weight
        if weighted_total <= 0:
            return 0.0
        return weighted_hit / weighted_total

    def _proximity_score(self, query_terms: list[str], doc_idx: int) -> float:
        if not query_terms:
            return 0.0
        unique_terms = sorted(set(query_terms))
        if len(unique_terms) == 1:
            return 1.0 if unique_terms[0] in self._doc_term_positions[doc_idx] else 0.0

        positions_map = self._doc_term_positions[doc_idx]
        events: list[tuple[int, str]] = []
        for term in unique_terms:
            for pos in positions_map.get(term, []):
                events.append((pos, term))
        if not events:
            return 0.0
        events.sort(key=lambda item: item[0])

        need = len(unique_terms)
        best_span = math.inf
        best_coverage = 0.0
        seen: defaultdict[str, int] = defaultdict(int)
        covered = 0
        left = 0

        for right, (_, right_term) in enumerate(events):
            if seen[right_term] == 0:
                covered += 1
            seen[right_term] += 1

            while left <= right:
                left_term = events[left][1]
                if seen[left_term] <= 1:
                    break
                seen[left_term] -= 1
                left += 1

            best_coverage = max(best_coverage, covered / need)
            if covered == need:
                left_pos = events[left][0]
                right_pos = events[right][0]
                best_span = min(best_span, float(right_pos - left_pos + 1))

        if math.isinf(best_span):
            return 0.35 * best_coverage

        compactness = min(1.0, need / best_span)
        return 0.6 * best_coverage + 0.4 * compactness

    def search(self, query: str, top_k: int, *, allowed_books: set[str] | None = None) -> list[RetrievalHit]:
        with self._lock:
            if not self._fitted:
                return []

            top_k = max(1, min(top_k, settings.retrieval_top_k_max))
            core_terms = sorted(self.preprocessor.core_query_terms(query))
            expanded_terms = sorted(set(self.preprocessor.tokenize(query)))
            exact_terms = sorted(set(self.preprocessor.tokenize_exact(query)))
            if not core_terms:
                return []

            min_core_terms = 1 if len(core_terms) <= 2 else max(1, math.ceil(len(core_terms) * 0.5))
            query_norm = query.lower().replace("ё", "е").strip()
            allowed = None if allowed_books is None else set(allowed_books)

            raw_scores: list[float] = []
            for i, chunk in enumerate(self._chunks):
                if allowed is not None and chunk.book not in allowed:
                    raw_scores.append(0.0)
                    continue

                core_score, core_matches = self._bm25_score(core_terms, i)
                expanded_score, expanded_matches = self._bm25_score(expanded_terms, i)
                exact_matches = sum(1 for t in exact_terms if t in self._doc_exact_set[i])
                exact_ratio = exact_matches / len(exact_terms) if exact_terms else 0.0

                coverage = self._weighted_coverage(core_terms, i)
                proximity = self._proximity_score(core_terms, i)

                if core_matches == 0 and expanded_matches == 0 and exact_matches == 0:
                    raw_scores.append(0.0)
                    continue

                if len(core_terms) >= 3 and core_matches < min_core_terms and coverage < 0.45 and exact_matches == 0:
                    raw_scores.append(0.0)
                    continue

                if len(core_terms) == 1:
                                                                                  
                                                                                  
                    lexical = 0.45 * core_score + 0.55 * expanded_score
                else:
                    lexical = 0.72 * core_score + 0.28 * expanded_score
                structure = 0.57 * coverage + 0.43 * proximity
                adjusted = lexical * (0.45 + 0.55 * structure)
                adjusted += 0.16 * exact_ratio

                text_norm = chunk.text.lower().replace("ё", "е")
                if query_norm and query_norm in text_norm:
                    adjusted += 0.18

                if core_matches > 0 and len(core_terms) > 1:
                    adjusted += 0.05 * (core_matches / len(core_terms))

                if coverage < 0.2 and exact_ratio == 0:
                    adjusted *= 0.75

                raw_scores.append(adjusted)

            max_score = max(raw_scores) if raw_scores else 0.0
            if max_score <= 0:
                return []

            normalized_scores = [s / max_score for s in raw_scores]
            ranked_idx = sorted(range(len(normalized_scores)), key=lambda i: normalized_scores[i], reverse=True)[:top_k]
            hits = [RetrievalHit(chunk=self._chunks[i], score=float(normalized_scores[i])) for i in ranked_idx]

            threshold = settings.min_search_score
            if len(core_terms) == 1:
                threshold = min(settings.min_search_score, 0.03)
            return [h for h in hits if h.score >= threshold]

    def sentence_scores(self, query: str, sentences: list[str]) -> list[tuple[str, float]]:
        if not sentences:
            return []

        query_core_terms = sorted(self.preprocessor.core_query_terms(query))
        query_expanded_terms = self.preprocessor.tokenize(query)
        query_exact_terms = sorted(set(self.preprocessor.tokenize_exact(query)))
        query_exact_set = set(query_exact_terms)

        if not query_core_terms and not query_expanded_terms:
            return [(s, 0.0) for s in sentences]

        def cosine(lhs: Counter[str], rhs: Counter[str]) -> float:
            lhs_norm = math.sqrt(sum(v * v for v in lhs.values())) or 1.0
            rhs_norm = math.sqrt(sum(v * v for v in rhs.values())) or 1.0
            dot = sum(lhs[t] * rhs.get(t, 0) for t in lhs)
            return dot / (lhs_norm * rhs_norm)

        q_core_tf = Counter(query_core_terms)
        q_expanded_tf = Counter(query_expanded_terms)
        query_norm = query.lower().replace("ё", "е").strip()

        scored: list[tuple[str, float]] = []
        for sentence in sentences:
            sentence_core_terms = [
                t for t in self.preprocessor.tokenize(sentence, include_synonyms=False) if not self.preprocessor.is_morph_token(t)
            ]
            s_core_tf = Counter(sentence_core_terms)
            s_expanded_tf = Counter(self.preprocessor.tokenize(sentence))
            core_cos = cosine(q_core_tf, s_core_tf) if q_core_tf else 0.0
            expanded_cos = cosine(q_expanded_tf, s_expanded_tf) if q_expanded_tf else 0.0

            sentence_exact = set(self.preprocessor.tokenize_exact(sentence))
            exact_overlap = 0.0
            if query_exact_terms:
                exact_overlap = len(sentence_exact & query_exact_set) / len(query_exact_terms)

            phrase_bonus = 0.08 if query_norm and query_norm in sentence.lower().replace("ё", "е") else 0.0
            if len(query_core_terms) == 1:
                score = 0.48 * core_cos + 0.42 * expanded_cos + 0.10 * exact_overlap + phrase_bonus
            else:
                score = 0.62 * core_cos + 0.28 * expanded_cos + 0.10 * exact_overlap + phrase_bonus
            scored.append((sentence, score))

        return sorted(scored, key=lambda x: x[1], reverse=True)
