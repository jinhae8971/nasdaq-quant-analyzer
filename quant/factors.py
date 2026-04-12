"""
factors.py — 6대 퀀트 팩터 계산 엔진

팩터:
  1. Momentum (20%) — RSI, MACD, 수익률, 이격도
  2. Value    (20%) — Forward P/E, P/S, P/B, EV/EBITDA, PEG, FCF Yield
  3. Quality  (15%) — ROE, ROA, Gross Margin, D/E, FCF/Revenue
  4. Growth   (15%) — Revenue Growth, EPS Growth, Forward EPS Growth
  5. Risk     (15%) — Beta, 변동성, MDD, Sharpe, Sortino
  6. Flow     (15%) — 거래량 변화, 기관 보유율, Short Interest
"""
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        v = float(val)
        if v != v:  # NaN check
            return None
        return v
    except (ValueError, TypeError):
        return None


class FactorEngine:
    """종목별 6대 팩터 원시값을 추출한다."""

    def compute(self, stock: Dict) -> Dict[str, Dict[str, Optional[float]]]:
        tech = stock.get("technical", {})
        return {
            "momentum": self._momentum(stock, tech),
            "value":    self._value(stock),
            "quality":  self._quality(stock),
            "growth":   self._growth(stock),
            "risk":     self._risk(stock, tech),
            "flow":     self._flow(stock),
        }

    # ── Momentum ──────────────────────────────────────────────────────────

    def _momentum(self, stock: Dict, tech: Dict) -> Dict[str, Optional[float]]:
        return {
            "rsi":              _safe_float(tech.get("rsi")),
            "macd_histogram":   _safe_float(tech.get("histogram")),
            "return_1m":        _safe_float(tech.get("return_1m")),
            "return_3m":        _safe_float(tech.get("return_3m")),
            "return_6m":        _safe_float(tech.get("return_6m")),
            "return_1y":        _safe_float(tech.get("return_1y")),
            "ma20_diff_pct":    _safe_float(tech.get("ma20_diff_pct")),
            "ma60_diff_pct":    _safe_float(tech.get("ma60_diff_pct")),
            "ma200_diff_pct":   _safe_float(tech.get("ma200_diff_pct")),
            "from_52w_high":    _safe_float(tech.get("from_52w_high_pct")),
            "bb_pct_b":         _safe_float(tech.get("pct_b")),
        }

    # ── Value ─────────────────────────────────────────────────────────────

    def _value(self, stock: Dict) -> Dict[str, Optional[float]]:
        # Forward P/E가 낮을수록 좋음 → 역수 변환은 scoring에서 처리
        fcf = _safe_float(stock.get("fcf_per_share"))
        price = _safe_float(stock.get("price"))
        mcap = _safe_float(stock.get("market_cap"))
        fcf_yield = None
        if fcf is not None and mcap is not None and mcap > 0:
            fcf_yield = fcf / mcap * 100  # %

        return {
            "forward_pe":   _safe_float(stock.get("forward_pe")),
            "trailing_pe":  _safe_float(stock.get("trailing_pe")),
            "ps_ratio":     _safe_float(stock.get("ps_ratio")),
            "pb_ratio":     _safe_float(stock.get("pb_ratio")),
            "ev_ebitda":    _safe_float(stock.get("ev_ebitda")),
            "peg_ratio":    _safe_float(stock.get("peg_ratio")),
            "fcf_yield":    round(fcf_yield, 2) if fcf_yield is not None else None,
            "dividend_yield": _safe_float(stock.get("dividend_yield")),
        }

    # ── Quality ───────────────────────────────────────────────────────────

    def _quality(self, stock: Dict) -> Dict[str, Optional[float]]:
        roe = _safe_float(stock.get("roe"))
        roa = _safe_float(stock.get("roa"))
        return {
            "roe":              round(roe * 100, 2) if roe is not None else None,
            "roa":              round(roa * 100, 2) if roa is not None else None,
            "gross_margin":     self._pct(_safe_float(stock.get("gross_margin"))),
            "operating_margin": self._pct(_safe_float(stock.get("operating_margin"))),
            "profit_margin":    self._pct(_safe_float(stock.get("profit_margin"))),
            "debt_to_equity":   _safe_float(stock.get("debt_to_equity")),
            "current_ratio":    _safe_float(stock.get("current_ratio")),
        }

    # ── Growth ────────────────────────────────────────────────────────────

    def _growth(self, stock: Dict) -> Dict[str, Optional[float]]:
        rev_g = _safe_float(stock.get("revenue_growth"))
        earn_g = _safe_float(stock.get("earnings_growth"))
        fwd_eps = _safe_float(stock.get("forward_eps"))
        trail_eps = _safe_float(stock.get("trailing_eps"))
        eps_growth_fwd = None
        if fwd_eps is not None and trail_eps is not None and trail_eps != 0:
            eps_growth_fwd = round((fwd_eps - trail_eps) / abs(trail_eps) * 100, 2)
        return {
            "revenue_growth":    round(rev_g * 100, 2) if rev_g is not None else None,
            "earnings_growth":   round(earn_g * 100, 2) if earn_g is not None else None,
            "eps_growth_fwd":    eps_growth_fwd,
            "analyst_target_upside": self._target_upside(stock),
        }

    # ── Risk ──────────────────────────────────────────────────────────────

    def _risk(self, stock: Dict, tech: Dict) -> Dict[str, Optional[float]]:
        return {
            "beta":         _safe_float(stock.get("beta")),
            "vol_20d":      _safe_float(tech.get("vol_20d")),
            "vol_60d":      _safe_float(tech.get("vol_60d")),
            "max_drawdown": _safe_float(tech.get("max_drawdown")),
            "sharpe":       _safe_float(tech.get("sharpe")),
            "sortino":      _safe_float(tech.get("sortino")),
        }

    # ── Flow ──────────────────────────────────────────────────────────────

    def _flow(self, stock: Dict) -> Dict[str, Optional[float]]:
        inst = _safe_float(stock.get("held_pct_institutions"))
        return {
            "volume_ratio":           _safe_float(stock.get("volume_ratio")),
            "institutional_pct":      round(inst * 100, 2) if inst is not None else None,
            "short_ratio":            _safe_float(stock.get("short_ratio")),
            "short_pct_float":        self._pct(_safe_float(stock.get("short_pct_float"))),
            "insider_pct":            self._pct(_safe_float(stock.get("held_pct_insiders"))),
        }

    # ── 헬퍼 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _pct(val: Optional[float]) -> Optional[float]:
        if val is None:
            return None
        return round(val * 100, 2)

    @staticmethod
    def _target_upside(stock: Dict) -> Optional[float]:
        target = _safe_float(stock.get("target_mean_price"))
        price = _safe_float(stock.get("price"))
        if target is not None and price is not None and price > 0:
            return round((target - price) / price * 100, 2)
        return None
