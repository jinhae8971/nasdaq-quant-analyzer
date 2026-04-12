"""
pipeline.py — 전체 파���프라인 오케스트레이터

실행 순서:
  1. 데이터 수집
  2. 퀀트 스크리닝 (팩터 → 스코어 → 시그널)
  3. Top N 종목 AI 심층분석
  4. 결과 저장 (JSON)
  5. Telegram 알림 (선택)
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import anthropic

from agents import MomentumAgent, FundamentalAgent, RiskAgent, FlowAgent
from orchestrator.synthesizer import Synthesizer
from quant.screener import QuantScreener
from scripts.collect_data import collect_market_data

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, model: str = "claude-sonnet-4-20250514", top_n: int = 10):
        self.model = model
        self.top_n = top_n
        self.root = Path(__file__).resolve().parent.parent
        self.screener = QuantScreener()

    def run(self) -> Dict:
        """전체 파이프라인 실행"""
        logger.info("=" * 60)
        logger.info("NASDAQ Quant Analyzer — 파이프라인 시작")
        logger.info("=" * 60)

        # 1. 데이터 수집
        market_data = collect_market_data()

        # 2. 퀀트 스크리닝
        logger.info("\n[Pipeline] 퀀트 스크리닝 시작...")
        screener_result = self.screener.run(market_data)

        # 3. AI 심층분석 대상 선정
        targets = self.screener.get_deep_analysis_targets(screener_result, self.top_n)
        logger.info(f"\n[Pipeline] AI 심층분석 대상: {targets}")

        # 4. AI 멀티에이전트 분석
        ai_analysis = {}
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if api_key and targets:
            try:
                client = anthropic.Anthropic(api_key=api_key)
                agents = [
                    MomentumAgent(client, self.model),
                    FundamentalAgent(client, self.model),
                    RiskAgent(client, self.model),
                    FlowAgent(client, self.model),
                ]
                synthesizer = Synthesizer(agents, client, self.model)

                for ticker in targets:
                    stock_data = market_data["stocks"].get(ticker, {})
                    quant_data = screener_result["stocks"].get(ticker, {})
                    if stock_data:
                        logger.info(f"\n[AI] {ticker} 심층분석 시작...")
                        result = synthesizer.analyze_stock(ticker, stock_data, quant_data)
                        ai_analysis[ticker] = result
            except Exception as e:
                logger.error(f"[AI] 에이전트 분석 실패: {e}")
        else:
            if not api_key:
                logger.warning("[AI] ANTHROPIC_API_KEY 미설정 — AI 분석 건너뜀")

        # 5. 최종 보고서 구성
        report = self._build_report(market_data, screener_result, ai_analysis)

        # 6. 저장
        self._save_report(report)

        # 7. Telegram 알림
        self._send_telegram(report)

        logger.info("\n" + "=" * 60)
        logger.info("파이프라인 완료!")
        logger.info("=" * 60)

        return report

    def _build_report(self, market_data: Dict, screener_result: Dict, ai_analysis: Dict) -> Dict:
        # AI 분석 결과를 각 종목에 병합
        stocks = screener_result.get("stocks", {})
        for ticker, analysis in ai_analysis.items():
            if ticker in stocks:
                stocks[ticker]["ai_analysis"] = analysis

        return {
            "date":         market_data.get("date"),
            "generated_at": datetime.now().isoformat(),
            "indices":      market_data.get("indices", {}),
            "sectors":      market_data.get("sectors", {}),
            "stocks":       stocks,
            "top_picks":    screener_result.get("top_picks", {}),
            "summary":      screener_result.get("summary", {}),
        }

    def _save_report(self, report: Dict):
        # docs/data/daily_report.json (대시보드 연동)
        docs_dir = self.root / "docs" / "data"
        docs_dir.mkdir(parents=True, exist_ok=True)
        report_path = docs_dir / "daily_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"[Save] {report_path}")

        # data/history/ (일별 이력)
        history_dir = self.root / "data" / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        date_str = report.get("date", datetime.now().strftime("%Y-%m-%d"))
        hist_path = history_dir / f"{date_str}.json"
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"[Save] {hist_path}")

    def _send_telegram(self, report: Dict):
        token = os.getenv("TELEGRAM_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            logger.info("[Telegram] 설정 없음 — 알림 건너뜀")
            return

        import requests

        top_buy = report.get("top_picks", {}).get("buy", [])
        top_sell = report.get("top_picks", {}).get("sell", [])
        summary = report.get("summary", {})

        lines = [
            "📊 NASDAQ Quant Analyzer",
            f"📅 {report.get('date', 'N/A')}",
            f"📈 분석 종목: {summary.get('total_stocks', 0)}개",
            f"📊 평균 퀀트스코어: {summary.get('avg_score', 0)}",
            "",
            "🟢 Top Buy Picks:",
        ]
        for s in top_buy[:3]:
            lines.append(f"  • {s['ticker']} ({s['composite_score']:.0f}점, {s['grade']})")

        if top_sell:
            lines.append("\n🔴 Top Sell Picks:")
            for s in top_sell[:3]:
                lines.append(f"  • {s['ticker']} ({s['composite_score']:.0f}점, {s['grade']})")

        pages_url = f"https://jinhae8971.github.io/nasdaq-quant-analyzer/"
        lines.append(f"\n🔗 {pages_url}")

        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": "\n".join(lines)},
                timeout=10,
            )
            logger.info("[Telegram] 알림 전송 완료")
        except Exception as e:
            logger.error(f"[Telegram] 전송 실패: {e}")
