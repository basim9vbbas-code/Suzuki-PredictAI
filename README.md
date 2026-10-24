# Suzuki PredictAI - Market Intelligence

University project: a market-intelligence module (economic indicators, news
sentiment, scenario analysis) intended to later feed a vehicle-demand
forecasting system.

## Status
- [x] Stage 0: Environment and project setup (in progress)
- [ ] Stage 1: Data-source research
- [ ] Stage 2-9: planned

## Setup (Windows, PowerShell)
    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python scripts\verify_setup.py

## Limitations
Economic indicators and sentiment alone do not predict exact vehicle sales.
Any scenario output is an assumption or uncalibrated estimate until validated
against historical demand data.