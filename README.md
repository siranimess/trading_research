# Multi-Agent Trading Research System

A portfolio-ready research project that uses **LangChain Deep Agents + OpenAI** to propose, critique, and coordinate systematic trading experiments, while a deterministic Python engine independently scores every strategy.

The design separates **LLM reasoning** from **numerical evaluation**:

- LLM agents propose and critique strategy specifications.
- The backtester is deterministic; the LLM never calculates its own score.
- Candidate evaluations receive validation data only.
- Holdout data stays sealed until a champion has been selected and frozen.
- The champion is hashed with SHA-256 before the holdout is opened.
- The holdout is evaluated only once per experiment directory.

> Educational/research use only. This is not investment advice or a production trading system.

## Architecture

```text
Strategy Engineer ----\
                       \
Research Critic --------> Python experiment orchestrator
                       /             |
Coordinator ----------/              v
                             validated StrategySpec JSON
                                      |
                                      v
                             isolated subprocess scorer
                                      |
                                      v
                              deterministic Pandas
                                  backtester
                                      |
                                      v
                          fixed validation gates/ranking
                                      |
                                      v
                            frozen champion + SHA-256
                                      |
                                      v
                             one holdout evaluation
```

## Project structure

```text
multi_agent_trading_research/
├── .env.example
├── .gitignore
├── config.yaml
├── pyproject.toml
├── requirements.txt
├── README.md
├── src/trading_research/
│   ├── __init__.py
│   ├── agents.py
│   ├── backtest.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── pipeline.py
│   ├── prompts.py
│   ├── report.py
│   ├── sandbox.py
│   ├── sandbox_worker.py
│   └── schemas.py
└── tests/
    ├── test_backtest.py
    ├── test_gates.py
    └── test_split.py
```

## Install

### Windows / Git Bash

```bash
cd multi_agent_trading_research

python -m venv .venv
source .venv/Scripts/activate

python -m pip install --upgrade pip
pip install -e .
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

## API configuration

Copy the example:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
OPENAI_API_KEY=your_key_here
TRADING_LLM_MODEL=openai:gpt-5.6-luna
```

The API key exists only in the parent orchestration process. Every scoring subprocess receives a scrubbed environment.

## First run: no API key and no internet needed

```bash
trading-research run --mode mock --synthetic
```

This runs the entire architecture with deterministic mock agents and deterministic synthetic OHLCV data.

## Real run: ETF data + LangChain Deep Agents + OpenAI

```bash
trading-research run --mode llm
```

The default config downloads adjusted daily data for SPY, QQQ, and IWM with `yfinance`.

## Tests

```bash
pytest -q
```

## How the system works

### 1. Market data

`data.py` either downloads U.S. equity ETF data or generates deterministic synthetic data for offline testing. Data is split chronologically into:

- train,
- validation,
- sealed holdout.

Random splitting is deliberately avoided because time-series data must preserve temporal order.

### 2. Strategy Engineer

The Strategy Engineer returns JSON only. It can choose one of three strategy families:

- `trend`
- `mean_reversion`
- `breakout`

Example:

```json
{
  "name": "trend_20_100",
  "family": "trend",
  "fast_window": 20,
  "slow_window": 100,
  "max_position": 1.0
}
```

Pydantic validates the specification. The LLM never sends executable Python into the backtester.

### 3. Deterministic backtesting

The backtester applies the same logic to every candidate:

```text
asset_return     = close.pct_change()
held_position    = signal.shift(1)
turnover         = abs(position - previous_position)
transaction_cost = turnover * cost_rate
strategy_return  = held_position * asset_return - transaction_cost
```

The signal shift means today's close can only affect tomorrow's position.

### 4. Subprocess scoring boundary

Each candidate is scored in a separate worker process. The worker gets:

- one copied CSV,
- one validated strategy JSON file,
- a clean temporary working directory,
- a scrubbed environment,
- a timeout.

The candidate scorer is never given the holdout path or API credentials.

This is a process-level isolation boundary, not a kernel security sandbox. The stronger design decision is that **model-generated code is never executed at all**. If you later decide to run arbitrary generated Python, use Docker/gVisor/Firecracker.

### 5. Research Critic

The critic gets only validation results and checks for:

- weak Sharpe,
- large drawdowns,
- excessive turnover,
- parameter fragility,
- unnecessary complexity.

It is explicitly told not to request holdout results.

### 6. Coordinator

The coordinator receives validation results plus the critic's review and decides whether the next round should:

- refine,
- diversify,
- stop.

Python still enforces the maximum number of rounds.

### 7. Fixed selection rule

A strategy must pass all gates from `config.yaml`:

```yaml
selection:
  min_validation_sharpe: 0.50
  max_validation_drawdown: 0.30
  max_annual_turnover: 30.0
```

Passing strategies are ranked deterministically by:

1. higher validation Sharpe,
2. smaller absolute max drawdown,
3. lower annual turnover,
4. alphabetical strategy name.

The agents cannot change this rule after seeing results.

### 8. Champion freeze

The selected champion is serialized canonically and SHA-256 hashed. The JSON and hash are written to `frozen/`.

The holdout evaluator re-computes the hash before scoring. If the strategy has changed, evaluation is refused.

### 9. Single holdout evaluation

After holdout scoring, the pipeline writes:

```text
HOLDOUT_CONSUMED
```

A second holdout evaluation inside the same experiment directory is refused.

## What each file does

- `config.py`: typed YAML configuration.
- `schemas.py`: Pydantic strategy/result objects.
- `data.py`: yfinance download, synthetic demo data, chronological split.
- `backtest.py`: strategy signals and deterministic metrics.
- `sandbox.py`: subprocess launcher, clean environment, timeout.
- `sandbox_worker.py`: minimal child-process entry point.
- `prompts.py`: role instructions for all three agents.
- `agents.py`: Deep Agents implementation plus offline mock agents.
- `pipeline.py`: rounds, validation, gates, champion freeze, holdout.
- `report.py`: JSON/Markdown experiment report.
- `cli.py`: command-line interface.

## Resume wording this code supports

**Multi-Agent Trading Research System — LLM Orchestration, Deep Agents, Python, Pandas, OpenAI API**

- Architected a multi-agent trading research system with LangChain Deep Agents, coordinating strategy-engineer, research-critic, and coordinator agents to develop systematic strategies against historical U.S. equity ETF data.
- Built a deterministic Pandas backtesting engine and isolated subprocess scorer that evaluates every agent-generated strategy independently while withholding credentials and holdout data.
- Implemented a staged experiment pipeline with fixed Sharpe, drawdown, and turnover gates, then freezes the selected champion before a single holdout evaluation.

## Limitations

This is intentionally a research system, not a broker/execution engine. Daily-close backtests omit many real-world effects such as spread modeling, market impact, taxes, borrow cost, execution uncertainty, survivorship concerns, and institutional-quality data cleaning. Historical performance does not imply future profitability.
