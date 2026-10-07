from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from dotenv import load_dotenv

from .prompts import (
    COORDINATOR_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    ENGINEER_SYSTEM_PROMPT,
)
from .schemas import CoordinatorDecision, Critique, EvaluationResult, StrategySpec


def _result_text(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if not messages:
        raise RuntimeError("Deep Agent returned no messages.")

    content = messages[-1].content
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)

    return str(content)


def _extract_json(text: str) -> Any:
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


class AgentSuite(ABC):
    @abstractmethod
    def propose(
        self,
        *,
        round_index: int,
        count: int,
        history: list[EvaluationResult],
        direction: CoordinatorDecision | None,
    ) -> list[StrategySpec]:
        raise NotImplementedError

    @abstractmethod
    def critique(self, results: list[EvaluationResult]) -> Critique:
        raise NotImplementedError

    @abstractmethod
    def coordinate(
        self,
        results: list[EvaluationResult],
        critique: Critique,
    ) -> CoordinatorDecision:
        raise NotImplementedError


class DeepAgentSuite(AgentSuite):
    def __init__(self, model: str | None = None) -> None:
        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Add it to .env or use --mode mock."
            )

        from deepagents import create_deep_agent

        self.model = model or os.getenv(
            "TRADING_LLM_MODEL",
            "openai:gpt-5.6-luna",
        )

        self.engineer = create_deep_agent(
            model=self.model,
            tools=[],
            system_prompt=ENGINEER_SYSTEM_PROMPT,
            name="strategy-engineer",
        )
        self.critic = create_deep_agent(
            model=self.model,
            tools=[],
            system_prompt=CRITIC_SYSTEM_PROMPT,
            name="research-critic",
        )
        self.coordinator = create_deep_agent(
            model=self.model,
            tools=[],
            system_prompt=COORDINATOR_SYSTEM_PROMPT,
            name="coordinator",
        )

    @staticmethod
    def _invoke(agent: Any, prompt: str) -> Any:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        return _extract_json(_result_text(result))

    def propose(
        self,
        *,
        round_index: int,
        count: int,
        history: list[EvaluationResult],
        direction: CoordinatorDecision | None,
    ) -> list[StrategySpec]:
        previous = [
            {
                "strategy": row.strategy.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
                "metrics": row.metrics.model_dump(mode="json"),
                "passed_gates": row.passed_gates,
                "gate_failures": row.gate_failures,
            }
            for row in history[-6:]
        ]

        prompt = json.dumps(
            {
                "round": round_index,
                "required_count": count,
                "previous_validation_results": previous,
                "coordinator_direction": (
                    direction.model_dump(mode="json") if direction else None
                ),
                "instruction": (
                    "Return exactly required_count distinct strategies. "
                    "Use only StrategySpec fields."
                ),
            },
            indent=2,
        )

        payload = self._invoke(self.engineer, prompt)
        strategies = [
            StrategySpec.model_validate(item)
            for item in payload.get("strategies", [])
        ]

        if len(strategies) != count:
            raise ValueError(
                f"Engineer returned {len(strategies)} strategies; expected {count}."
            )

        return strategies

    def critique(self, results: list[EvaluationResult]) -> Critique:
        prompt = json.dumps(
            [
                {
                    "strategy": row.strategy.model_dump(
                        mode="json",
                        exclude_none=True,
                    ),
                    "metrics": row.metrics.model_dump(mode="json"),
                    "passed_gates": row.passed_gates,
                    "gate_failures": row.gate_failures,
                }
                for row in results
            ],
            indent=2,
        )
        return Critique.model_validate(self._invoke(self.critic, prompt))

    def coordinate(
        self,
        results: list[EvaluationResult],
        critique: Critique,
    ) -> CoordinatorDecision:
        prompt = json.dumps(
            {
                "validation_results": [
                    row.model_dump(mode="json") for row in results
                ],
                "critic": critique.model_dump(mode="json"),
            },
            indent=2,
        )
        return CoordinatorDecision.model_validate(
            self._invoke(self.coordinator, prompt)
        )


class MockAgentSuite(AgentSuite):
    BASE_CANDIDATES = [
        {
            "name": "trend_10_60",
            "family": "trend",
            "fast_window": 10,
            "slow_window": 60,
        },
        {
            "name": "trend_20_120",
            "family": "trend",
            "fast_window": 20,
            "slow_window": 120,
        },
        {
            "name": "reversion_20_1p5",
            "family": "mean_reversion",
            "lookback": 20,
            "entry_z": 1.5,
            "exit_z": 0.0,
        },
        {
            "name": "breakout_55_20",
            "family": "breakout",
            "lookback": 55,
            "exit_lookback": 20,
        },
        {
            "name": "trend_50_200",
            "family": "trend",
            "fast_window": 50,
            "slow_window": 200,
        },
        {
            "name": "reversion_60_2p0",
            "family": "mean_reversion",
            "lookback": 60,
            "entry_z": 2.0,
            "exit_z": 0.0,
        },
        {
            "name": "breakout_100_40",
            "family": "breakout",
            "lookback": 100,
            "exit_lookback": 40,
        },
        {
            "name": "trend_30_150",
            "family": "trend",
            "fast_window": 30,
            "slow_window": 150,
        },
        {
            "name": "reversion_40_1p75",
            "family": "mean_reversion",
            "lookback": 40,
            "entry_z": 1.75,
            "exit_z": 0.0,
        },
    ]

    def propose(
        self,
        *,
        round_index: int,
        count: int,
        history: list[EvaluationResult],
        direction: CoordinatorDecision | None,
    ) -> list[StrategySpec]:
        start = ((round_index - 1) * count) % len(self.BASE_CANDIDATES)
        selected: list[StrategySpec] = []

        for offset in range(count):
            raw = dict(
                self.BASE_CANDIDATES[
                    (start + offset) % len(self.BASE_CANDIDATES)
                ]
            )
            raw["name"] = f"r{round_index}_{raw['name']}"
            selected.append(StrategySpec.model_validate(raw))

        return selected

    def critique(self, results: list[EvaluationResult]) -> Critique:
        passing = sum(row.passed_gates for row in results)
        highest_turnover = max(
            row.metrics.annual_turnover for row in results
        )

        return Critique(
            summary=(
                f"{passing}/{len(results)} candidates passed the fixed gates."
            ),
            risks=[
                f"Highest annual turnover: {highest_turnover:.2f}.",
                "Validation performance may be regime-specific.",
            ],
            suggestions=[
                "Keep parameters interpretable.",
                "Do not change gates after seeing validation results.",
            ],
        )

    def coordinate(
        self,
        results: list[EvaluationResult],
        critique: Critique,
    ) -> CoordinatorDecision:
        passing = [row for row in results if row.passed_gates]

        if passing:
            best = max(passing, key=lambda row: row.metrics.sharpe)
            return CoordinatorDecision(
                direction="refine",
                rationale=(
                    f"A candidate passed; continue around "
                    f"{best.strategy.family.value}."
                ),
                preferred_family=best.strategy.family,
            )

        return CoordinatorDecision(
            direction="diversify",
            rationale="No candidate passed; diversify next round.",
            preferred_family=None,
        )
