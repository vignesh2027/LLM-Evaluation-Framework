"""
Cost calculation metric for LLM API calls.

Maintains a pricing table for major providers and computes exact USD
costs from token counts. Prices are per-1M-tokens (input / output).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelPricing:
    """Pricing for a single model in USD per 1M tokens."""

    input_per_1m: float
    output_per_1m: float


# Pricing table (USD per 1M tokens, as of 2025-01)
_PRICING: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o": ModelPricing(5.00, 15.00),
    "gpt-4o-mini": ModelPricing(0.15, 0.60),
    "gpt-4-turbo": ModelPricing(10.00, 30.00),
    "gpt-4": ModelPricing(30.00, 60.00),
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50),
    "o1": ModelPricing(15.00, 60.00),
    "o1-mini": ModelPricing(3.00, 12.00),
    # Anthropic Claude
    "claude-3-5-sonnet-20241022": ModelPricing(3.00, 15.00),
    "claude-3-5-haiku-20241022": ModelPricing(0.80, 4.00),
    "claude-3-opus-20240229": ModelPricing(15.00, 75.00),
    "claude-3-sonnet-20240229": ModelPricing(3.00, 15.00),
    "claude-3-haiku-20240307": ModelPricing(0.25, 1.25),
    # Google Gemini
    "gemini/gemini-1.5-pro": ModelPricing(3.50, 10.50),
    "gemini/gemini-1.5-flash": ModelPricing(0.075, 0.30),
    "gemini/gemini-2.0-flash-exp": ModelPricing(0.00, 0.00),
    # Mistral
    "mistral/mistral-large-latest": ModelPricing(4.00, 12.00),
    "mistral/mistral-small-latest": ModelPricing(1.00, 3.00),
    "mistral/open-mistral-7b": ModelPricing(0.25, 0.25),
    # Meta Llama (via Together)
    "together_ai/meta-llama/Llama-3-70b-chat-hf": ModelPricing(0.90, 0.90),
    "together_ai/meta-llama/Llama-3-8b-chat-hf": ModelPricing(0.20, 0.20),
    # Default fallback
    "default": ModelPricing(1.00, 2.00),
}


class CostMetric:
    """Calculate USD cost from token usage and model name."""

    def calculate(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Return exact USD cost for the given token counts and model."""
        pricing = self._get_pricing(model)
        input_cost = (input_tokens / 1_000_000) * pricing.input_per_1m
        output_cost = (output_tokens / 1_000_000) * pricing.output_per_1m
        return input_cost + output_cost

    def estimate_run_cost(
        self,
        model: str,
        num_samples: int,
        avg_input_tokens: int = 150,
        avg_output_tokens: int = 200,
    ) -> float:
        """Estimate total cost for a benchmark run before executing it."""
        per_sample = self.calculate(model, avg_input_tokens, avg_output_tokens)
        return per_sample * num_samples

    def _get_pricing(self, model: str) -> ModelPricing:
        if model in _PRICING:
            return _PRICING[model]
        # Partial match fallback (e.g., "gpt-4o-2024-05-13" → "gpt-4o")
        for key in _PRICING:
            if key in model or model in key:
                return _PRICING[key]
        return _PRICING["default"]

    def cost_per_1k_tokens(
        self, model: str, total_cost: float, total_tokens: int
    ) -> float:
        """Return cost per 1,000 tokens given total cost and total token count."""
        if total_tokens == 0:
            return 0.0
        return (total_cost / total_tokens) * 1000

    def get_all_pricing(self) -> dict[str, ModelPricing]:
        """Return the full pricing table (excludes default entry)."""
        return {k: v for k, v in _PRICING.items() if k != "default"}
