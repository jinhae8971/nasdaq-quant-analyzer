"""
collect_data.py — NASDAQ 종목 데이터 수집 및 기술적 지표 계산

수집 항목:
  - NASDAQ Composite, S&P500, VIX 지수
  - ~50개 NASDAQ 종목 일봉(1년) + 재무 정보
  - 섹터 ETF 등락률
  - 기술적 지표: RSI, MACD, 볼린저밴드, 이동평균
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ─── 유니버스 정의 ─────────────────────────────────────────────────────────────

INDEX_TICKERS = {
    "nasdaq": "^IXIC",
    "sp500":  "^GSPC",
    "vix":    "^VIX",
}

SECTOR_ETFS = {
    "Technology":    "XLK",
    "Healthcare":    "XLV",
    "Financials":    "XLF",
    "Consumer Disc": "XLY",
    "Communication": "XLC",
    "Industrials":   "XLI",
    "Staples":       "XLP",
    "Energy":        "XLE",
    "Utilities":     "XLU",
    "Real Estate":   "XLRE",
    "Materials":     "XLB",
}

STOCK_UNIVERSE = {
    # Tech
    "AAPL":  {"name": "Apple",          "sector": "Technology"},
    "MSFT":  {"name": "Microsoft",      "sector": "Technology"},
    "NVDA":  {"name": "NVIDIA",         "sector": "Technology"},
    "GOOGL": {"name": "Alphabet",       "sector": "Technology"},
    "META":  {"name": "Meta Platforms",  "sector": "Technology"},
    "AMZN":  {"name": "Amazon",         "sector": "Technology"},
    "AVGO":  {"name": "Broadcom",       "sector": "Technology"},
    "TSLA":  {"name": "Tesla",          "sector": "Technology"},
    "AMD":   {"name": "AMD",            "sector": "Technology"},
    "INTC":  {"name": "Intel",          "sector": "Technology"},
    "QCOM":  {"name": "Qualcomm",       "sector": "Technology"},
    "CRM":   {"name": "Salesforce",     "sector": "Technology"},
    "ADBE":  {"name": "Adobe",          "sector": "Technology"},
    "NFLX":  {"name": "Netflix",        "sector": "Communication"},
    "ORCL":  {"name": "Oracle",         "sector": "Technology"},
    "AMAT":  {"name": "Applied Materials", "sector": "Technology"},
    "MU":    {"name": "Micron",         "sector": "Technology"},
    "LRCX":  {"name": "Lam Research",   "sector": "Technology"},
    "KLAC":  {"name": "KLA Corp",       "sector": "Technology"},
    "MRVL":  {"name": "Marvell",        "sector": "Technology"},
    "SNPS":  {"name": "Synopsys",       "sector": "Technology"},
    "CDNS":  {"name": "Cadence",        "sector": "Technology"},
    # Cybersecurity
    "PANW":  {"name": "Palo Alto",      "sector": "Technology"},
    "CRWD":  {"name": "CrowdStrike",    "sector": "Technology"},
    "FTNT":  {"name": "Fortinet",       "sector": "Technology"},
    # Biotech/Health
    "AMGN":  {"name": "Amgen",          "sector": "Healthcare"},
    "GILD":  {"name": "Gilead",         "sector": "Healthcare"},
    "ISRG":  {"name": "Intuitive Surgical", "sector": "Healthcare"},
    "VRTX":  {"name": "Vertex Pharma",  "sector": "Healthcare"},
    "REGN":  {"name": "Regeneron",      "sector": "Healthcare"},
    "MRNA":  {"name": "Moderna",        "sector": "Healthcare"},
    "DXCM":  {"name": "DexCom",         "sector": "Healthcare"},
    # Fintech/Payment
    "PYPL":  {"name": "PayPal",         "sector": "Financials"},
    "INTU":  {"name": "Intuit",         "sector": "Financials"},
    "COIN":  {"name": "Coinbase",       "sector": "Financials"},
    # Consumer
    "COST":  {"name": "Costco",         "sector": "Consumer Disc"},
    "PEP":   {"name": "PepsiCo",        "sector": "Staples"},
    "SBUX":  {"name": "Starbucks",      "sector": "Consumer Disc"},
    "ABNB":  {"name": "Airbnb",         "sector": "Consumer Disc"},
    "BKNG":  {"name": "Booking",        "sector": "Consumer Disc"},
    # Communication
    "CMCSA": {"name": "Comcast",        "sector": "Communication"},
    "TMUS":  {"name": "T-Mobile",       "sector": "Communication"},
    # Industrial
    "HON":   {"name": "Honeywell",      "sector": "Industrials"},
    "GE":    {"name": "GE Aerospace",   "sector": "Industrials"},
    "CAT":   {"name": "Caterpillar",    "sector": "Industrials"},
    # Energy
    "XOM":   {"name": "ExxonMobil",     "sector": "Energy"},
    "CEG":   {"name": "Constellation Energy", "sector": "Energy"},
}


# ─── 기술적 지표 계산 ─────────────────────────────────────────────────────────

def calc_rsi(series: pd.Series, period: int = 14) -> Optional[float]:
    if len(series) < period + 1:
        return None
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    val = (100 - 100 / (1 + rs)).iloc[-1]
    return round(float(val), 2) if not np.isnan(val) else None


def calc_macd(series: pd.Series) -> Dict[str, Optional[float]]:
    if len(series) < 35:
        return {"macd": None, "signal": None, "histogram": None}
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return {
        "macd":      round(float(macd.iloc[-1]), 4),
        "signal":    round(float(signal.iloc[-1]), 4),
        "histogram": round(float(histogram.iloc[-1]), 4),
    }


def calc_bollinger(series: pd.Series, period: int = 20) -> Dict[str, Optional[float]]:
    if len(series) < period:
        return {"upper": None, "middle": None, "lower": None, "pct_b": None}
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    pct_b = (series - lower) / (upper - lower)
    return {
        "upper":  round(float(upper.iloc[-1]), 2),
        "middle": round(float(sma.iloc[-1]), 2),
        "lower":  round(float(lower.iloc[-1]), 2),
        "pct_b":  round(float(pct_b.iloc[-1]), 4) if not np.isnan(pct_b.iloc[-1]) else None,
    }


def calc_moving_averages(series: pd.Series) -> Dict[str, Optional[float]]:
    result = {}
    current = float(series.iloc[-1])
    for period in [5, 20, 60, 120, 200]:
        if len(series) >= period:
            ma = float(series.rolling(period).mean().iloc[-1])
            result[f"ma{period}"] = round(ma, 2)
            result[f"ma{period}_diff_pct"] = round((current - ma) / ma * 100, 2)
        else:
            result[f"ma{period}"] = None
            result[f"ma{period}_diff_pct"] = None
    return result


def calc_volatility(series: pd.Series) -> Dict[str, Optional[float]]:
    if len(series) < 60:
        return {"vol_20d": None, "vol_60d": None}
    returns = series.pct_change().dropna()
    vol_20 = float(returns.tail(20).std() * np.sqrt(252) * 100)
    vol_60 = float(returns.tail(60).std() * np.sqrt(252) * 100)
    return {
        "vol_20d": round(vol_20, 2),
        "vol_60d": round(vol_60, 2),
    }


def calc_returns(series: pd.Series) -> Dict[str, Optional[float]]:
    current = float(series.iloc[-1])
    result = {}
    for label, days in [("1w", 5), ("1m", 21), ("3m", 63), ("6m", 126), ("1y", 252)]:
        if len(series) > days:
            past = float(series.iloc[-(days + 1)])
            result[f"return_{label}"] = round((current - past) / past * 100, 2)
        else:
            result[f"return_{label}"] = None
    # 52주 고저
    if len(series) >= 252:
        h = float(series.tail(252).max())
        l = float(series.tail(252).min())
        result["high_52w"] = round(h, 2)
        result["low_52w"] = round(l, 2)
        result["from_52w_high_pct"] = round((current - h) / h * 100, 2)
    return result


def calc_max_drawdown(series: pd.Series, window: int = 252) -> Optional[float]:
    if len(series) < window:
        return None
    tail = series.tail(window)
    peak = tail.expanding(min_periods=1).max()
    drawdown = (tail - peak) / peak
    return round(float(drawdown.min()) * 100, 2)


def calc_sharpe_sortino(series: pd.Series, risk_free_rate: float = 0.04) -> Dict[str, Optional[float]]:
    if len(series) < 252:
        return {"sharpe": None, "sortino": None}
    returns = series.tail(252).pct_change().dropna()
    excess = returns - risk_free_rate / 252
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else None
    downside = returns[returns < 0]
    sortino = None
    if len(downside) > 0 and downside.std() > 0:
        sortino = float(excess.mean() / downside.std() * np.sqrt(252))
    return {
        "sharpe":  round(sharpe, 2) if sharpe is not None else None,
        "sortino": round(sortino, 2) if sortino is not None else None,
    }


# ─── 종목 데이터 수집 ─────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str, meta: Dict, period: str = "1y") -> Optional[Dict]:
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        if hist.empty or len(hist) < 20:
            logger.warning(f"  [{ticker}] 데이터 부족 (rows={len(hist)})")
            return None

        closes = hist["Close"]
        volumes = hist["Volume"]
        info = {}
        try:
            info = t.info or {}
        except Exception:
            logger.warning(f"  [{ticker}] info 조회 실패")

        price = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2]) if len(closes) > 1 else price
        change_pct = round((price - prev_close) / prev_close * 100, 2)

        # 거래량 분석
        avg_vol_20 = float(volumes.tail(20).mean()) if len(volumes) >= 20 else float(volumes.mean())
        last_vol = float(volumes.iloc[-1])
        vol_ratio = round(last_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 1.0

        return {
            "ticker":     ticker,
            "name":       meta["name"],
            "sector":     meta["sector"],
            "price":      round(price, 2),
            "change_pct": change_pct,
            "volume":     int(last_vol),
            "avg_volume_20d": int(avg_vol_20),
            "volume_ratio": vol_ratio,
            # 재무 지표 (yfinance info)
            "market_cap":       info.get("marketCap"),
            "forward_pe":       info.get("forwardPE"),
            "trailing_pe":      info.get("trailingPE"),
            "ps_ratio":         info.get("priceToSalesTrailing12Months"),
            "pb_ratio":         info.get("priceToBook"),
            "ev_ebitda":        info.get("enterpriseToEbitda"),
            "peg_ratio":        info.get("pegRatio"),
            "dividend_yield":   info.get("dividendYield"),
            "beta":             info.get("beta"),
            "roe":              info.get("returnOnEquity"),
            "roa":              info.get("returnOnAssets"),
            "gross_margin":     info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "profit_margin":    info.get("profitMargins"),
            "debt_to_equity":   info.get("debtToEquity"),
            "current_ratio":    info.get("currentRatio"),
            "revenue_growth":   info.get("revenueGrowth"),
            "earnings_growth":  info.get("earningsGrowth"),
            "fcf_per_share":    info.get("freeCashflow"),
            "revenue":          info.get("totalRevenue"),
            "short_ratio":      info.get("shortRatio"),
            "short_pct_float":  info.get("shortPercentOfFloat"),
            "held_pct_insiders":      info.get("heldPercentInsiders"),
            "held_pct_institutions":  info.get("heldPercentInstitutions"),
            "forward_eps":      info.get("forwardEps"),
            "trailing_eps":     info.get("trailingEps"),
            "target_mean_price": info.get("targetMeanPrice"),
            "recommendation":   info.get("recommendationKey"),
            "analyst_count":    info.get("numberOfAnalystOpinions"),
            # 기술적 지표
            "technical": {
                "rsi":        calc_rsi(closes),
                **calc_macd(closes),
                **calc_bollinger(closes),
                **calc_moving_averages(closes),
                **calc_volatility(closes),
                **calc_returns(closes),
                "max_drawdown": calc_max_drawdown(closes),
                **calc_sharpe_sortino(closes),
            },
        }
    except Exception as e:
        logger.error(f"  [{ticker}] 수집 실패: {e}")
        return None


def fetch_index_data() -> Dict[str, Any]:
    result = {}
    for name, ticker in INDEX_TICKERS.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if hist.empty:
                continue
            close = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else close
            result[name] = {
                "close":      round(close, 2),
                "change_pct": round((close - prev) / prev * 100, 2),
            }
        except Exception as e:
            logger.error(f"  [지수 {name}] 수집 실패: {e}")
    return result


def fetch_sector_data() -> Dict[str, Any]:
    result = {}
    for name, ticker in SECTOR_ETFS.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if hist.empty:
                continue
            close = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else close
            result[name] = {
                "ticker":     ticker,
                "close":      round(close, 2),
                "change_pct": round((close - prev) / prev * 100, 2),
            }
        except Exception as e:
            logger.error(f"  [섹터 {name}] 수집 실패: {e}")
    return result


# ─── 메인 수집 함수 ────────────────────────────────────────────────────────────

def collect_market_data() -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("NASDAQ Quant Analyzer — 데이터 수집 시작")
    logger.info("=" * 60)

    # 1. 지수
    logger.info("[1/3] 지수 데이터 수집...")
    indices = fetch_index_data()

    # 2. 섹터 ETF
    logger.info("[2/3] 섹터 ETF 데이터 수집...")
    sectors = fetch_sector_data()

    # 3. 개별 종목
    logger.info(f"[3/3] 개별 종목 데이터 수집 ({len(STOCK_UNIVERSE)}개)...")
    stocks = {}
    for i, (ticker, meta) in enumerate(STOCK_UNIVERSE.items(), 1):
        logger.info(f"  ({i}/{len(STOCK_UNIVERSE)}) {ticker}...")
        data = fetch_stock_data(ticker, meta)
        if data:
            stocks[ticker] = data

    logger.info(f"수집 완료: 지수 {len(indices)}개, 섹터 {len(sectors)}개, 종목 {len(stocks)}개")

    return {
        "date":       datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().isoformat(),
        "indices":    indices,
        "sectors":    sectors,
        "stocks":     stocks,
    }
