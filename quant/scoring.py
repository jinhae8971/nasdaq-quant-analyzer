"""
scoring.py — 종합 퀀트 스코어링 엔진

각 팩터 원시값 → 섹터 내 백분위(0-100) 정규화 → 가중 합산 → 등급
"""
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 팩터별 가중치
FACTOR_WEIGHTS = {
    "momentum": 0.20,
    "value":    0.20,
    "quality":  0.15,
    "growth":   0.15,
    "risk":     0.15,
    "flow":     0.15,
}

# 각 서브팩터의 방향 (higher_is_better)
# True = 높을수록 좋음, False = 낮을수록 좋음
METRIC_DIRECTION = {
    # Momentum
    "rsi":              "neutral",   # 50 근처가 가장 좋고, 극단값은 나쁨
    "macd_histogram":   "higher",
    "return_1m":        "higher",
    "return_3m":        "higher",
    "return_6m":        "higher",
    "return_1y":        "higher",
    "ma20_diff_pct":    "higher",
    "ma60_diff_pct":    "higher",
    "ma200_diff_pct":   "higher",
    "from_52w_high":    "higher",    # 0에 가까울수록 좋음 (음수)
    "bb_pct_b":         "neutral",   # 0.5 근처 이상적
    # Value (낮을수록 좋음)
    "forward_pe":       "lower",
    "trailing_pe":      "lower",
    "ps_ratio":         "lower",
    "pb_ratio":         "lower",
    "ev_ebitda":        "lower",
    "peg_ratio":        "lower",
    "fcf_yield":        "higher",
    "dividend_yield":   "higher",
    # Quality
    "roe":              "higher",
    "roa":              "higher",
    "gross_margin":     "higher",
    "operating_margin": "higher",
    "profit_margin":    "higher",
    "debt_to_equity":   "lower",
    "current_ratio":    "higher",
    # Growth
    "revenue_growth":       "higher",
    "earnings_growth":      "higher",
    "eps_growth_fwd":       "higher",
    "analyst_target_upside": "higher",
    # Risk (낮은 리스크가 좋음 → Sharpe/Sortino는 높을수록)
    "beta":         "lower",
    "vol_20d":      "lower",
    "vol_60d":      "lower",
    "max_drawdown": "higher",    # max_drawdown은 음수, 0에 가까울수록 좋음
    "sharpe":       "higher",
    "sortino":      "higher",
    # Flow
    "volume_ratio":      "higher",
    "institutional_pct": "higher",
    "short_ratio":       "lower",
    "short_pct_float":   "lower",
    "insider_pct":       "neutral",  # 극단값 모두 안 좋음
}

GRADE_THRESHOLDS = [
    (90, "S"),
    (75, "A"),
    (60, "B"),
    (45, "C"),
    (30, "D"),
    (0,  "F"),
]


def assign_grade(score: float) -> str:
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


class ScoringEngine:
    """
    팩터 원시값을 받아 섹터 내 백분위로 정규화하고 종합 스코어를 산출한다.
    """

    def score_all(self, factors_by_ticker: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Args:
            factors_by_ticker: {ticker: {factor_name: {metric: value}}}
        Returns:
            {ticker: {
                "composite_score": float,
                "grade": str,
                "factor_scores": {factor: float},
                "rank": int,
                "sector_rank": int,
            }}
        """
        tickers = list(factors_by_ticker.keys())
        if not tickers:
            return {}

        # 1. 섹터별 그룹핑을 위한 메타데이터 (stocks_data에서 가져옴)
        # scoring은 factor값만 받으므로, 전체 비교로 백분위 계산

        # 2. 각 팩터별 서브스코어 계산
        factor_scores = {}  # {ticker: {factor_name: score}}
        for factor_name in FACTOR_WEIGHTS:
            metrics = self._collect_metrics(factors_by_ticker, factor_name)
            scores = self._percentile_rank(metrics, factor_name)
            for ticker, score in scores.items():
                factor_scores.setdefault(ticker, {})[factor_name] = score

        # 3. 종합 스코어 계산
        results = {}
        for ticker in tickers:
            fs = factor_scores.get(ticker, {})
            composite = sum(
                fs.get(f, 50) * w for f, w in FACTOR_WEIGHTS.items()
            )
            composite = round(composite, 1)
            results[ticker] = {
                "composite_score": composite,
                "grade":           assign_grade(composite),
                "factor_scores":   {k: round(v, 1) for k, v in fs.items()},
            }

        # 4. 전체 랭킹
        sorted_tickers = sorted(results, key=lambda t: results[t]["composite_score"], reverse=True)
        for rank, ticker in enumerate(sorted_tickers, 1):
            results[ticker]["rank"] = rank
            results[ticker]["total_count"] = len(sorted_tickers)

        return results

    def _collect_metrics(
        self, factors_by_ticker: Dict, factor_name: str
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """특정 팩터의 모든 종목 메트릭을 수집한다."""
        result = {}
        for ticker, factors in factors_by_ticker.items():
            result[ticker] = factors.get(factor_name, {})
        return result

    def _percentile_rank(
        self, metrics_by_ticker: Dict[str, Dict], factor_name: str
    ) -> Dict[str, float]:
        """
        각 메트릭을 백분위로 변환한 후 팩터 내 평균 = 팩터 스코어
        """
        tickers = list(metrics_by_ticker.keys())
        if not tickers:
            return {}

        # 모든 메트릭 이름 수집
        all_metric_names = set()
        for m in metrics_by_ticker.values():
            all_metric_names.update(m.keys())

        # 메트릭별 백분위
        metric_percentiles = {}  # {metric: {ticker: percentile}}
        for metric in all_metric_names:
            vals = []
            for ticker in tickers:
                v = metrics_by_ticker[ticker].get(metric)
                if v is not None:
                    vals.append((ticker, v))

            if len(vals) < 2:
                # 데이터 부족 → 모두 50점
                metric_percentiles[metric] = {t: 50.0 for t in tickers}
                continue

            direction = METRIC_DIRECTION.get(metric, "higher")

            # 값 기준 정렬
            if direction == "lower":
                vals.sort(key=lambda x: x[1])  # 작은 값이 높은 순위
            elif direction == "neutral":
                # 중앙값(50 또는 0.5)에 가까울수록 높은 점수
                if metric == "rsi":
                    vals.sort(key=lambda x: abs(x[1] - 50))
                elif metric == "bb_pct_b":
                    vals.sort(key=lambda x: abs(x[1] - 0.5))
                else:
                    vals.sort(key=lambda x: x[1])
            else:  # higher
                vals.sort(key=lambda x: x[1], reverse=True)  # 큰 값이 높은 순위

            n = len(vals)
            pct_map = {}
            for rank, (t, _) in enumerate(vals):
                pct_map[t] = round((1 - rank / max(n - 1, 1)) * 100, 1)

            # 값이 없는 종목은 50점
            for t in tickers:
                if t not in pct_map:
                    pct_map[t] = 50.0

            metric_percentiles[metric] = pct_map

        # 팩터 스코어 = 메트릭 백분위의 평균
        factor_scores = {}
        for ticker in tickers:
            pcts = [
                metric_percentiles[m].get(ticker, 50.0)
                for m in all_metric_names
                if m in metric_percentiles
            ]
            factor_scores[ticker] = sum(pcts) / len(pcts) if pcts else 50.0

        return factor_scores
