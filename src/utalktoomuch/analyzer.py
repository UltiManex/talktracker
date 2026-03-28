from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable


STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
    "your",
    "we",
    "they",
    "i",
}


@dataclass(slots=True)
class SpeakerSegment:
    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None


class ConversationAnalyzer:
    def assign_speakers(
        self,
        transcript: Iterable[TranscriptSegment],
        diarization: Iterable[SpeakerSegment],
    ) -> list[TranscriptSegment]:
        diarization_segments = list(diarization)
        enriched: list[TranscriptSegment] = []

        for t_segment in transcript:
            overlaps: dict[str, float] = defaultdict(float)
            for d_segment in diarization_segments:
                overlap = min(t_segment.end, d_segment.end) - max(t_segment.start, d_segment.start)
                if overlap > 0:
                    overlaps[d_segment.speaker] += overlap

            speaker = max(overlaps, key=overlaps.get) if overlaps else "Unknown"
            enriched.append(
                TranscriptSegment(
                    start=t_segment.start,
                    end=t_segment.end,
                    text=t_segment.text.strip(),
                    speaker=speaker,
                )
            )

        return enriched

    def speaking_share(self, transcript: Iterable[TranscriptSegment]) -> dict[str, float]:
        speaking_time: dict[str, float] = defaultdict(float)
        total = 0.0

        for segment in transcript:
            duration = max(0.0, segment.end - segment.start)
            speaking_time[segment.speaker or "Unknown"] += duration
            total += duration

        if total == 0:
            return {speaker: 0.0 for speaker in speaking_time}

        return {speaker: (duration / total) * 100 for speaker, duration in speaking_time.items()}

    def summarize(self, transcript: Iterable[TranscriptSegment]) -> tuple[str, dict[str, str]]:
        full_text = " ".join(segment.text.strip() for segment in transcript if segment.text.strip())
        overall_summary = self._summary_from_text(full_text)

        speaker_buckets: dict[str, list[str]] = defaultdict(list)
        for segment in transcript:
            if segment.text.strip():
                speaker_buckets[segment.speaker or "Unknown"].append(segment.text)

        speaker_focus = {
            speaker: self._focus_from_text(" ".join(texts)) for speaker, texts in speaker_buckets.items()
        }
        return overall_summary, speaker_focus

    def _summary_from_text(self, text: str) -> str:
        if not text:
            return "No conversation content available yet."

        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        if not sentences:
            return "Conversation captured, but no full sentences were detected."

        return ". ".join(sentences[:3]) + ("." if not sentences[0].endswith(".") else "")

    def _focus_from_text(self, text: str) -> str:
        tokens = [
            token.strip(".,!?;:\"'()[]{}")
            for token in text.lower().split()
            if token.strip(".,!?;:\"'()[]{}")
        ]
        keyword_counts = Counter(token for token in tokens if token not in STOPWORDS and len(token) > 2)
        if not keyword_counts:
            return "General conversation and short responses."

        top_words = [word for word, _ in keyword_counts.most_common(3)]
        return f"Primarily focused on: {', '.join(top_words)}."
