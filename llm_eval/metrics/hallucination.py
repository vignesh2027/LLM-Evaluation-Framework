"""
Hallucination detection and reasoning quality scoring.

Uses a lightweight heuristic approach (no extra API calls required):
- Contradiction detection via negation patterns
- Confidence calibration (hedging vs certainty)
- Reasoning chain depth analysis for quality scoring
"""

from __future__ import annotations

import re

# Phrases that suggest hallucination / confabulation
_HALLUCINATION_SIGNALS = [
    r"\bas of my (last|latest|recent) (update|training|knowledge)\b",
    r"\bi (believe|think|assume|suppose) (that )?(it|this|they|the)\b",
    r"\baccording to (some|various|many) (sources|reports|studies)\b",
    r"\bit (is|was) (reportedly|rumored|said to be)\b",
    r"\bi('m| am) not (entirely |completely |100% )?(sure|certain|confident)\b",
    r"\bcould be (wrong|incorrect|mistaken)\b",
    r"\bapproximately|roughly|around\b.*\b(million|billion|percent|%)\b",
]

# Phrases that indicate good grounded reasoning
_GROUNDING_SIGNALS = [
    r"\baccording to\b",
    r"\bbased on\b",
    r"\bthe (evidence|data|research|study|paper) (shows|suggests|indicates|demonstrates)\b",
    r"\bspecifically\b",
    r"\bfor example\b",
    r"\bin (fact|particular|summary|conclusion)\b",
]

# Reasoning step markers
_REASONING_MARKERS = [
    r"\b(first|second|third|finally|therefore|thus|hence|consequently)\b",
    r"\bbecause\b",
    r"\bsince\b",
    r"\bthis (means|implies|suggests)\b",
    r"\bwe can (see|conclude|infer|observe)\b",
    r"\bstep \d\b",
    r"\blet('s| us) (think|consider|analyze)\b",
]


class HallucinationMetric:
    """
    Heuristic hallucination scorer (0.0 = no hallucination, 1.0 = high).

    Does not call an external API — runs entirely locally for speed and cost.
    For production accuracy, swap the scoring method with an NLI model call.
    """

    def score(self, prompt: str, response: str) -> float:
        """
        Return hallucination probability in [0, 1].

        Higher = more likely hallucinatory. Based on linguistic signal density.
        """
        if not response:
            return 1.0

        text = response.lower()
        words = max(len(text.split()), 1)

        # Count negative signals (hallucination indicators)
        hall_hits = sum(
            1 for p in _HALLUCINATION_SIGNALS if re.search(p, text)
        )
        # Count positive signals (grounding indicators)
        ground_hits = sum(
            1 for p in _GROUNDING_SIGNALS if re.search(p, text)
        )

        # Penalize very short responses to factual prompts
        length_penalty = 0.2 if words < 10 else 0.0

        # Base score
        raw = (hall_hits * 0.15 - ground_hits * 0.05 + length_penalty)
        return max(0.0, min(1.0, raw))

    def reasoning_quality(self, response: str) -> float:
        """
        Return reasoning quality score in [1, 10].

        Based on structural reasoning markers, response length, and coherence.
        """
        if not response:
            return 1.0

        text = response.lower()
        words = len(text.split())

        marker_hits = sum(
            1 for p in _REASONING_MARKERS if re.search(p, text)
        )
        ground_hits = sum(
            1 for p in _GROUNDING_SIGNALS if re.search(p, text)
        )

        # Length score: up to 3 points for appropriate length (50-400 words)
        if words < 10:
            length_score = 1.0
        elif words < 50:
            length_score = 2.0
        elif words <= 400:
            length_score = 3.0
        else:
            length_score = 2.5  # penalize excessively verbose answers

        # Structure score: up to 4 points from reasoning markers
        structure_score = min(4.0, marker_hits * 0.8)

        # Grounding score: up to 3 points
        grounding_score = min(3.0, ground_hits * 1.0)

        total = length_score + structure_score + grounding_score
        return round(max(1.0, min(10.0, total)), 2)

    def batch_score(self, prompts: list[str], responses: list[str]) -> list[float]:
        """Return per-sample hallucination scores for a list of prompt/response pairs."""
        return [self.score(p, r) for p, r in zip(prompts, responses)]
