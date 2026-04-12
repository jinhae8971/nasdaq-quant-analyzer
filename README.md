# NASDAQ Quant Analyzer

6대 팩터 기반 나스닥 종목 퀀트 분석 + AI 멀티에이전��� 투자 인사이�� 대시보드

## Features

- **6-Factor Quant Model**: Momentum, Value, Quality, Growth, Risk, Flow
- **4 AI Agents**: MomentumBot, ValueHunter, RiskGuard, FlowTracker
- **Cross-Debate**: 에이전트 간 교차 반론을 통한 균형 잡힌 분석
- **50+ NASDAQ Stocks**: 자동 스크리닝 + Top 10 AI 심층분석
- **GitHub Pages Dashboard**: 프로 트레이딩 터미널 스타일

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key"
python scripts/run_pipeline.py
```

## Architecture

```
scripts/collect_data.py  → yfinance 데이터 수집
quant/factors.py         → 6대 팩터 계산
quant/scoring.py         → 섹터 내 백분위 스코어링
quant/signals.py         → 매매 시그널 생성
agents/*                 → 4 AI 에이전트 분석
orchestrator/pipeline.py → 전체 파이프라인
index.html               → GitHub Pages 대시보드
```
