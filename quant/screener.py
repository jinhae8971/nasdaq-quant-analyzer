"""
screener.py — 퀀트 스크리너 파이프라인

전체 흐름:
  collect_data → factors → scoring → signals → 최종 결과
"""
import logging
from typing import Dict, List, Tuple

from .factors import FactorEngine
from .scoring import ScoringEngine
from .signals import SignalGenerator

logger = logging.getLogger(__name__)


class QuantScreener:
    """퀀트 스크리닝 전체 파이프라인을 실행한다."""

    def __init__(self):
        self.factor_engine = FactorEngine()
        self.scoring_engine = ScoringEngine()
        self.signal_generator = SignalGenerator()

    def run(self, market_data: Dict) -> Dict:
        """
        Args:
            market_data: collect_market_data() 결과
        Returns:
            {
                "stocks": {ticker: {score, grade, signal, factors, alerts, ...}},
                "top_picks": {"buy": [...], "sell": [...]},
                "summary": {...},
            }
        """
        stocks = market_data.get("stocks", {})
        if not stocks:
            logger.warning("종목 데이터 없음")
            return {"stocks": {}, "top_picks": {"buy": [], "sell": []}, "summary": {}}

        # Phase 1: 팩터 계산
        logger.info(f"[Quant] Phase 1: {len(stocks)}개 종목 팩터 계산...")
        factors_by_ticker = {}
        for ticker, stock in stocks.items():
            factors_by_ticker[ticker] = self.factor_engine.compute(stock)

        # Phase 2: 스코어링
        logger.info("[Quant] Phase 2: 스코어링...")
        scores = self.scoring_engine.score_all(factors_by_ticker)

        # Phase 3: 시그널 생성
        logger.info("[Quant] Phase 3: 시그널 생성...")
        results = {}
        for ticker, stock in stocks.items():
            score_data = scores.get(ticker, {})
            factor_data = factors_by_ticker.get(ticker, {})
            signal_data = self.signal_generator.generate(score_data, factor_data, stock)

            results[ticker] = {
                "ticker":          ticker,
                "name":            stock.get("name", ticker),
                "sector":          stock.get("sector", ""),
                "price":           stock.get("price"),
                "change_pct":      stock.get("change_pct"),
                "market_cap":      stock.get("market_cap"),
                "composite_score": score_data.get("composite_score", 0),
                "grade":           score_data.get("grade", "F"),
                "rank":            score_data.get("rank", 0),
                "total_count":     score_data.get("total_count", 0),
                "factor_scores":   score_data.get("factor_scores", {}),
                "signal":          signal_data["signal"],
                "signal_info":     signal_data["signal_info"],
                "alerts":          signal_data["alerts"],
                "reasons":         signal_data["reasons"],
                "factors_raw":     factor_data,
            }

        # Phase 4: Top picks
        sorted_results = sorted(results.values(), key=lambda x: x["composite_score"], reverse=True)
        top_buy = [r for r in sorted_results if r["signal"] in ("STRONG_BUY", "BUY")][:5]
        top_sell = [r for r in reversed(sorted_results) if r["signal"] in ("STRONG_SELL", "SELL")][:5]

        # 전체 시그널 분포
        signal_dist = {}
        for r in results.values():
            sig = r["signal"]
            signal_dist[sig] = signal_dist.get(sig, 0) + 1

        # 섹터별 평균 스코어
        sector_scores = {}
        for r in results.values():
            sec = r["sector"]
            sector_scores.setdefault(sec, []).append(r["composite_score"])
        sector_avg = {
            sec: round(sum(vals) / len(vals), 1)
            for sec, vals in sector_scores.items()
        }

        summary = {
            "total_stocks":     len(results),
            "signal_distribution": signal_dist,
            "avg_score":        round(sum(r["composite_score"] for r in results.values()) / len(results), 1),
            "sector_avg_scores": sector_avg,
        }

        logger.info(f"[Quant] 완료: {len(results)}개 종목 분석, 시그널 분포: {signal_dist}")

        return {
            "stocks":    results,
            "top_picks": {"buy": top_buy, "sell": top_sell},
            "summary":   summary,
        }

    def get_deep_analysis_targets(self, screener_result: Dict, n: int = 10) -> List[str]:
        """AI 심층분석 대상 종목 선정 (상위5 + 하위5)"""
        stocks = screener_result.get("stocks", {})
        sorted_stocks = sorted(stocks.values(), key=lambda x: x["composite_score"], reverse=True)
        top = [s["ticker"] for s in sorted_stocks[:n // 2]]
        bottom = [s["ticker"] for s in sorted_stocks[-(n // 2):]]
        return top + bottom
