from __future__ import annotations

from dataclasses import dataclass
import re

from app.config import settings
from app.core.preprocess import TextPreprocessor
from app.core.retrieve import RetrievalHit, Retriever
from app.core.store import BookStore

_PRONOUN_WORDS = {
    "он", "она", "оно", "они",
    "его", "ее", "ему", "ей", "им", "ими", "их",
    "него", "нее", "нему", "ней", "нем", "ним", "ними", "них",
    "тот", "та", "то", "те", "этот", "эта", "это", "эти",
}
_NON_NAME_CAPITALIZED = {
    "и", "а", "но", "или", "как", "что", "кто", "где", "когда", "почему", "зачем",
    "это", "этот", "эта", "эти", "тот", "та", "те", "вот", "там", "только", "если",
    "к", "ко", "в", "во", "на", "по", "у", "о", "об", "от", "до", "за", "из", "с", "со",
    "его", "ее", "их", "все", "весь", "вся", "все",
    "автор", "герой", "герои", "героиня", "персонаж", "персонажи",
    "глава", "часть", "том", "книга", "пролог", "эпилог",
    "введение", "предисловие", "послесловие", "заключение",
    "действие", "картина", "сцена", "сцены", "явление",
    "господин", "госпожа", "князь", "княгиня", "граф", "графиня", "барон", "баронесса",
    "генерал", "полковник", "капитан", "майор", "лейтенант", "доктор", "профессор",
    "император", "царь",
}
_WEAK_ANCHOR_TERMS = {
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
_WEAK_ANCHOR_PREFIXES = (
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
_CAUSAL_RE = re.compile(
    r"\b(?:"
    r"потому\s+что|"
    r"поскольку|"
    r"так\s+как|"
    r"из-?за|"
    r"поэтому|"
    r"оттого(?:\s+что)?|"
    r"ибо|"
    r"вследствие|"
    r"ввиду|"
    r"благодаря(?:\s+тому)?(?:\s+что)?|"
    r"по\s+причине|"
    r"по\s+этой\s+причине|"
    r"в\s+результате"
    r")\b",
    re.IGNORECASE,
)
_CAUSAL_PAIR_RE = re.compile(r"\bчем\b.+\bтем\b", re.IGNORECASE)
_OPPOSITE_MOTION_HINT_RE = re.compile(
    r"(?:тороп|спеш|поспеш|поех|поед|уех|ехал|ехать|едет|едут|отъезж|отправ|мчал|умчал)",
    re.IGNORECASE,
)
_WHY_QUESTION_RE = re.compile(r"^(?:почему|зачем|отчего|по какой причине|из-?за чего)\b", re.IGNORECASE)


@dataclass(slots=True)
class AnswerResult:
    answer: str
    hits: list[RetrievalHit]
    message: str | None = None


@dataclass(slots=True)
class SentenceCandidate:
    text: str
    score: float
    token_set: set[str]
    hit_rank: int
    sentence_order: int
    exact_overlap: float = 0.0
    anchor_coverage: float = 0.0
    semantic_score: float = 0.0
    support_score: float = 0.0
    has_causal_cue: bool = False


class RagService:
    def __init__(self) -> None:
        self.preprocessor = TextPreprocessor()
        self.store = BookStore()
        self.retriever = Retriever(self.preprocessor)
        self.retriever.rebuild(self.store.all_chunks())

    def upload_book(self, filename: str, content: str) -> int:
        added = self.store.add_book(filename, content)
        self.retriever.rebuild(self.store.all_chunks())
        return added

    def search_snippets(self, query: str, top_k: int, *, allowed_books: set[str] | None = None) -> list[RetrievalHit]:
        return self.retriever.search(query=query, top_k=top_k, allowed_books=allowed_books)

    @staticmethod
    def _normalize_space(text: str) -> str:
        return " ".join(text.split())

    @classmethod
    def _normalize_answer_sentence(cls, text: str) -> str:
        return cls._normalize_space(text)

    def _sentence_token_set(self, sentence: str) -> set[str]:
        return set(self.preprocessor.tokenize(sentence, include_synonyms=False))

    @staticmethod
    def _is_weak_anchor_term(term: str) -> bool:
        return term in _WEAK_ANCHOR_TERMS or any(term.startswith(prefix) for prefix in _WEAK_ANCHOR_PREFIXES)

    @classmethod
    def _effective_anchor_terms(cls, meaningful_terms: set[str], core_terms: set[str]) -> set[str]:
        anchor_terms = meaningful_terms or core_terms
        if not anchor_terms:
            return set()
        strong_terms = {term for term in anchor_terms if not cls._is_weak_anchor_term(term)}
        return strong_terms if strong_terms else anchor_terms

    @staticmethod
    def _question_is_why(question: str) -> bool:
        q = " ".join(question.lower().replace("ё", "е").split())
        return bool(_WHY_QUESTION_RE.match(q))

    @staticmethod
    def _has_causal_cue(sentence: str) -> bool:
        normalized = sentence.lower().replace("ё", "е")
        return bool(_CAUSAL_RE.search(normalized) or _CAUSAL_PAIR_RE.search(normalized))

    @staticmethod
    def _is_refusal_term(term: str) -> bool:
        return term.startswith("отказ")

    def _query_action_infinitives(self, raw_query_exact: list[str]) -> tuple[set[str], list[str]]:
        stems: set[str] = set()
        surface: list[str] = []
        for token in raw_query_exact:
            normalized = self.preprocessor._normalize_token(token)
            if len(normalized) < 4 or not normalized.endswith("ть"):
                continue
            stems.add(self.preprocessor._stem(normalized))
            if normalized not in surface:
                surface.append(normalized)
        return stems, surface

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        if union == 0:
            return 0.0
        return inter / union

    def _select_diverse_sentences(self, candidates: list[SentenceCandidate], top_n: int) -> list[SentenceCandidate]:
        if not candidates or top_n <= 0:
            return []

        pool = list(candidates)
        selected: list[SentenceCandidate] = []
        mmr_lambda = 0.78

        while pool and len(selected) < top_n:
            best_idx = -1
            best_value = float("-inf")
            for idx, candidate in enumerate(pool):
                redundancy = 0.0
                if selected:
                    redundancy = max(self._jaccard(candidate.token_set, s.token_set) for s in selected)
                mmr = mmr_lambda * candidate.score - (1.0 - mmr_lambda) * redundancy
                if mmr > best_value:
                    best_value = mmr
                    best_idx = idx

            if best_idx < 0:
                break
            selected.append(pool.pop(best_idx))

        return selected

    @staticmethod
    def _truncate_answer(sentences: list[str], max_chars: int = 340) -> str:
        if not sentences:
            return ""
        selected: list[str] = []
        total = 0
        for sentence in sentences:
            clean = " ".join(sentence.split())
            if not clean:
                continue
            extra = len(clean) + (1 if selected else 0)
            if selected and total + extra > max_chars:
                break
            if not selected and len(clean) > max_chars:
                return clean[: max_chars - 3] + "..."
            selected.append(clean)
            total += extra
        answer = " ".join(selected)
        if answer and answer[-1] not in ".!?…»\"'":
            answer = answer.rstrip(",:;") + "..."
        if answer:
            answer = answer.lstrip(" \t\n\r,;:-")
            if answer and answer[0].islower():
                answer = answer[0].upper() + answer[1:]
        return answer

    @staticmethod
    def _is_fragmentary_sentence(sentence: str) -> bool:
        text = sentence.strip()
        if not text:
            return True
        lead = text.lstrip(" \t\n\r\"'«»()[]{}—-")
        if not lead:
            return True
        first = lead[0]
        if first in ",.;:!?":
            return True
        if first.islower():
            return True
                                                                                           
        return len(lead) < 14 and lead[-1] not in ".!?…"

    @staticmethod
    def _question_expects_person(question: str) -> bool:
        q = " ".join(question.lower().replace("ё", "е").split())
        return q == "кто" or q.startswith("кто ") or q.startswith("у кого ")

    @staticmethod
    def _first_word(text: str) -> str:
        for raw in text.strip().split():
            token = "".join(ch for ch in raw if ch.isalpha())
            if token:
                return token
        return ""

    @classmethod
    def _starts_with_pronoun(cls, sentence: str) -> bool:
        first = cls._first_word(sentence).lower().replace("ё", "е")
        return first in _PRONOUN_WORDS

    @classmethod
    def _name_like_tokens(cls, sentence: str) -> list[str]:
        words: list[str] = []
        for raw in sentence.strip().split():
            token = "".join(ch for ch in raw if ch.isalpha() or ch == "-")
            if token:
                words.append(token)
        names: list[str] = []
        for token in words:
            if len(token) < 3:
                continue
            if "-" in token:
                parts = [p for p in token.split("-") if p]
                if not parts:
                    continue
                token = parts[0]
                if len(token) < 3:
                    continue
            low = token.lower().replace("ё", "е")
            if low in _NON_NAME_CAPITALIZED:
                continue
            if low in _PRONOUN_WORDS:
                continue
            if len(low) >= 5 and low.endswith(("и", "ы")):
                continue
            if token[0].isupper() and token[1:].islower():
                names.append(token)
        return names

    @classmethod
    def _has_name_like_token(cls, sentence: str) -> bool:
        return bool(cls._name_like_tokens(sentence))

    @classmethod
    def _person_hints(cls, sentences: list[str], hits: list[RetrievalHit], *, limit: int = 4) -> list[str]:
        hints: list[str] = []
        seen: set[str] = set()

        def add_from(text: str) -> None:
            for token in cls._name_like_tokens(text):
                key = token.lower().replace("ё", "е")
                if key in seen:
                    continue
                seen.add(key)
                hints.append(token)
                if len(hints) >= limit:
                    return

        for sentence in sentences:
            add_from(sentence)
            if len(hints) >= limit:
                return hints
        for hit in hits:
            add_from(hit.chunk.text)
            if len(hints) >= limit:
                return hints
        return hints

    @staticmethod
    def _low_confidence_result(hits: list[RetrievalHit]) -> AnswerResult:
        return AnswerResult(
            answer="Точных оснований для ответа не найдено. Ниже приведены самые близкие цитаты.",
            hits=hits,
            message="low_confidence",
        )

    def _confidence_score(self, ranked: list[SentenceCandidate]) -> float:
        if not ranked:
            return 0.0
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        margin = top.score if second is None else max(0.0, top.score - second.score)
        confidence = (
            0.52 * max(0.0, top.score)
            + 0.24 * top.support_score
            + 0.16 * top.semantic_score
            + 0.08 * margin
        )
        if second is not None:
            overlap = self._jaccard(top.token_set, second.token_set)
            if overlap >= 0.45:
                confidence += 0.03
        return confidence

    def answer_from_hits(self, question: str, hits: list[RetrievalHit]) -> AnswerResult:
        if not hits:
            return AnswerResult(
                answer="В загруженных книгах не найдено релевантных фрагментов для ответа.",
                hits=[],
                message="not_found",
            )

        raw_query_exact = self.preprocessor.meaningful_exact_tokens(question)
        query_core = self.preprocessor.core_query_terms(question)
        query_meaningful = self.preprocessor.meaningful_query_terms(question)
        query_anchor = self._effective_anchor_terms(query_meaningful, query_core)
        query_exact: set[str] = set()
        for token in raw_query_exact:
            stem = self.preprocessor._stem(self.preprocessor._normalize_token(token))
            if not query_anchor or stem in query_anchor:
                query_exact.add(token)
        if not query_exact:
            query_exact = set(raw_query_exact)
        query_speech = self.preprocessor.speech_query_terms(question)
        expects_person = self._question_expects_person(question)
        is_why_question = self._question_is_why(question)
        is_weak_anchor_query = bool(query_anchor) and all(self._is_weak_anchor_term(term) for term in query_anchor)
        refusal_terms = {term for term in query_anchor if self._is_refusal_term(term)}
        action_inf_stems, action_inf_words = self._query_action_infinitives(raw_query_exact)

        anchor_query_text = " ".join(sorted(query_anchor))
        query_anchor_expanded = set(self.preprocessor.tokenize(anchor_query_text)) if anchor_query_text else set()
        query_anchor_expanded_non_morph = {t for t in query_anchor_expanded if not self.preprocessor.is_morph_token(t)}
        query_anchor_short_morph: set[str] = set()
        for term in query_anchor:
            if len(term) > 4:
                continue
            morph = self.preprocessor._morph_token(term)
            if morph:
                query_anchor_short_morph.add(morph)

        sentence_map: dict[str, SentenceCandidate] = {}
        opposite_hint_text = ""
        opposite_hint_score = float("-inf")

        for hit_rank, hit in enumerate(hits):
            sentences = self.preprocessor.split_sentences(hit.chunk.text)
            scored_sentences = self.retriever.sentence_scores(question, sentences)
            sentence_order_map = {
                self._normalize_answer_sentence(sentence): idx
                for idx, sentence in enumerate(sentences)
            }
            hit_token_set = self._sentence_token_set(hit.chunk.text)
            hit_has_anchor = bool(query_anchor and (hit_token_set & query_anchor))

            for sentence, semantic_score in scored_sentences:
                normalized = self._normalize_answer_sentence(sentence)
                if not normalized:
                    continue

                token_set = self._sentence_token_set(normalized)
                if not token_set:
                    continue

                anchor_overlap = len(token_set & query_anchor) if query_anchor else 0
                expanded_anchor_overlap = (
                    len(token_set & query_anchor_expanded_non_morph) if query_anchor_expanded_non_morph else 0
                )
                short_morph_overlap = len(token_set & query_anchor_short_morph) if query_anchor_short_morph else 0
                has_direct_anchor = anchor_overlap > 0 or expanded_anchor_overlap > 0 or short_morph_overlap > 0
                has_causal_cue = is_why_question and self._has_causal_cue(normalized)

                if query_anchor and not has_direct_anchor:
                    weak_anchor_context_ok = is_weak_anchor_query and hit_has_anchor and not self._is_fragmentary_sentence(normalized)
                    if not weak_anchor_context_ok and not has_causal_cue:
                        continue

                sentence_exact = set(self.preprocessor.tokenize_exact(normalized))
                exact_overlap = (len(sentence_exact & query_exact) / len(query_exact)) if query_exact else 0.0
                anchor_coverage = (anchor_overlap / len(query_anchor)) if query_anchor else 0.0
                expanded_coverage = (
                    (expanded_anchor_overlap / len(query_anchor_expanded_non_morph))
                    if query_anchor_expanded_non_morph
                    else 0.0
                )
                support_score = max(exact_overlap, anchor_coverage, expanded_coverage)

                if support_score == 0.0 and semantic_score < 0.34 and hit.score < 0.42:
                    continue

                speech_coverage = (len(token_set & query_speech) / len(query_speech)) if query_speech else 0.0
                fragment_penalty = 0.11 if self._is_fragmentary_sentence(normalized) else 0.0
                person_bonus = 0.0
                if expects_person and self._has_name_like_token(normalized):
                    person_bonus += 0.08
                if expects_person and self._starts_with_pronoun(normalized):
                    person_bonus -= 0.07
                causal_bonus = 0.0
                if is_why_question and has_causal_cue:
                    causal_bonus += 0.18
                elif is_why_question and len(query_anchor) >= 2:
                    causal_bonus -= 0.05

                echo_penalty = 0.0
                if is_why_question and query_anchor and not has_causal_cue:
                    if anchor_coverage >= 0.8 and len(token_set) <= len(query_anchor) + 4:
                        echo_penalty = 0.13

                combined_score = (
                    0.52 * semantic_score
                    + 0.28 * hit.score
                    + 0.14 * support_score
                    + 0.06 * speech_coverage
                    + person_bonus
                    + causal_bonus
                    - fragment_penalty
                    - echo_penalty
                )

                if is_why_question and action_inf_stems and (token_set & action_inf_stems):
                    text_norm = normalized.lower().replace("ё", "е")
                    if _OPPOSITE_MOTION_HINT_RE.search(text_norm) and combined_score > opposite_hint_score:
                        opposite_hint_score = combined_score
                        opposite_hint_text = normalized

                existing = sentence_map.get(normalized)
                if existing is None or combined_score > existing.score:
                    sentence_map[normalized] = SentenceCandidate(
                        text=normalized,
                        score=combined_score,
                        token_set=token_set,
                        hit_rank=hit_rank,
                        sentence_order=sentence_order_map.get(normalized, 0),
                        exact_overlap=exact_overlap,
                        anchor_coverage=anchor_coverage,
                        semantic_score=semantic_score,
                        support_score=support_score,
                        has_causal_cue=has_causal_cue,
                    )

        if not sentence_map:
            return self._low_confidence_result(hits)

        ranked = sorted(sentence_map.values(), key=lambda item: item.score, reverse=True)
        non_fragment_ranked = [item for item in ranked if not self._is_fragmentary_sentence(item.text)]
        if non_fragment_ranked:
            ranked = non_fragment_ranked

        if is_why_question and ranked:
            causal_ranked = [item for item in ranked if item.has_causal_cue]
            if causal_ranked and not ranked[0].has_causal_cue:
                ranked = causal_ranked + [item for item in ranked if item not in causal_ranked]

        if is_why_question and refusal_terms and action_inf_stems:
            refusal_action_ranked = [
                item
                for item in ranked
                if (item.token_set & refusal_terms) and (item.token_set & action_inf_stems)
            ]
            if refusal_action_ranked:
                ranked = refusal_action_ranked
            else:
                action_word = action_inf_words[0] if action_inf_words else "это делать"
                if opposite_hint_text:
                    return AnswerResult(
                        answer=f"В найденных фрагментах нет прямого подтверждения, что герой отказался {action_word}. "
                        f"Напротив: {opposite_hint_text}",
                        hits=hits,
                        message="low_confidence",
                    )
                return self._low_confidence_result(hits)

        if ranked and len(query_anchor) >= 2:
            min_anchor_overlap = 2 if len(query_anchor) >= 3 else 1
            if is_why_question:
                overlap_ranked = [
                    item
                    for item in ranked
                    if len(item.token_set & query_anchor) >= min_anchor_overlap or item.has_causal_cue
                ]
            else:
                overlap_ranked = [item for item in ranked if len(item.token_set & query_anchor) >= min_anchor_overlap]
            if overlap_ranked:
                ranked = overlap_ranked

        if not ranked:
            return self._low_confidence_result(hits)

        if ranked[0].score < 0.24:
            return self._low_confidence_result(hits)

        confidence = self._confidence_score(ranked)
        confidence_floor = 0.22 if is_why_question else 0.27
        if confidence < confidence_floor:
            return self._low_confidence_result(hits)

        if is_why_question and (not ranked[0].has_causal_cue) and ranked[0].support_score < 0.55:
            return self._low_confidence_result(hits)

        dynamic_floor = max(settings.min_answer_score, ranked[0].score * (0.58 if is_why_question else 0.52))
        filtered = [candidate for candidate in ranked if candidate.score >= dynamic_floor]
        if not filtered:
            filtered = [ranked[0]]

        if is_why_question:
            selected = [filtered[0]]
            for candidate in filtered[1:]:
                if len(selected) >= 2:
                    break
                if not candidate.has_causal_cue:
                    continue
                if self._jaccard(candidate.token_set, selected[0].token_set) >= 0.92:
                    continue
                selected.append(candidate)
            selected = sorted(selected, key=lambda item: (item.has_causal_cue, item.score), reverse=True)
        else:
            lead = filtered[0]
            selected = [lead]
            for candidate in filtered[1:]:
                if len(selected) >= 2:
                    break
                same_context = candidate.hit_rank == lead.hit_rank and candidate.sentence_order >= lead.sentence_order
                good_score = candidate.score >= max(lead.score * 0.74, 0.28)
                good_support = candidate.support_score >= 0.2 or candidate.semantic_score >= 0.44
                low_redundancy = self._jaccard(candidate.token_set, lead.token_set) < 0.90
                if same_context and good_score and good_support and low_redundancy:
                    selected.append(candidate)
            if len(selected) == 1 and len(filtered) > 1:
                diverse = self._select_diverse_sentences(filtered, 2)
                if diverse:
                    selected = sorted(diverse, key=lambda item: (item.hit_rank, item.sentence_order, -item.score))

        best = [candidate.text for candidate in selected if candidate.text]
        if not best:
            return self._low_confidence_result(hits)

        answer = self._truncate_answer(best, max_chars=680)
        if not answer:
            return self._low_confidence_result(hits)

        if expects_person and (self._starts_with_pronoun(answer) or not self._has_name_like_token(answer)):
            hints = self._person_hints(best, hits, limit=4)
            if hints:
                answer = f"По найденным фрагментам: {', '.join(hints)}. {answer}"

        return AnswerResult(answer=answer, hits=hits)

    def ask(self, question: str, top_k: int, *, allowed_books: set[str] | None = None) -> AnswerResult:
        hits = self.search_snippets(question, top_k=top_k, allowed_books=allowed_books)
        return self.answer_from_hits(question, hits)

    def stats(self) -> tuple[int, int]:
        return self.store.stats()
