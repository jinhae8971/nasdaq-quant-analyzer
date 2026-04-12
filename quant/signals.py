"""
signals.py — 매매 시그널 생성기

퀀트 스코어 + 팩터 조건 → 5단계 시그널 + 특수 경고
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SIGNAL_LEVELS = {
    "STRONG_BUY":  {"label": "Strong Buy",  "emoji": "🟢🟢", "color": "#00e676"},
    "BUY":         {"label": "Buy",         "emoji": "🟢",   "color": "#66bb6a"},
    "HOLD":        {"label": "Hold",        "emoji": "🟡",   "color": "#ffd740"},
    "SELL":        {"label": "Sell",        "emoji": "🔴",   "color": "#ff5252"},
    "STRONG_SELL": {"label": "Strong Sell", "emoji": "🔴🔴", "color": "#ff1744"},
}


class SignalGenerator:
    """퀀트 스코어와 팩터 데이터를 기반으로 매매 시그널을 생성한다."""

    def generate(
        self,
        score_data: Dict,
        factor_data: Dict,
        raw_stock: Dict,
    ) -> Dict:
        """
        Args:
            score_data: scoring.py 출력 (composite_score, factor_scores, grade 등)
            factor_data: factors.py 출력 (momentum, value 등 원시값)
            raw_stock: collect_data.py 출력 (기술적 지표 포함)
        Returns:
            {signal, signal_info, alerts, reasons}
        """
        composite = score_data.get("composite_score", 50)
        fs = score_data.get("factor_scores", {})
        tech = raw_stock.get("technical", {})

        # ── 메인 시그널 ────────────────────────────────────────────────────

        signal = self._determine_signal(composite, fs)

        # ── 시그널 근거 ────────────────────────────────────────────────────

        reasons = self._generate_reasons(composite, fs, factor_data, tech)

        # ── 특수 경고 ───────────────��──────────────────��───────────────────

        alerts = self._check_alerts(factor_data, tech, raw_stock)

        return {
            "signal":      signal,
            "signal_info": SIGNAL_LEVELS[signal],
            "alerts":      alerts,
            "reasons":     reasons,
        }

    def _determine_signal(self, composite: float, fs: Dict) -> str:
        momentum = fs.get("momentum", 50)
        value = fs.get("value", 50)
        quality = fs.get("quality", 50)
        risk = fs.get("risk", 50)

        # Strong Buy: 종합 75+, 2개 이상 팩터 70+
        if composite >= 75:
            above_70 = sum(1 for v in fs.values() if v >= 70)
            if above_70 >= 2:
                return "STRONG_BUY"

        # Buy: 종합 60+, 2개 이상 팩터 60+
        if composite >= 60:
            above_60 = sum(1 for v in fs.values() if v >= 60)
            if above_60 >= 2:
                return "BUY"

        # Strong Sell: 종합 25 미만
        if composite < 25:
            return "STRONG_SELL"

        # Sell: 종합 40 미만 또는 리스크 팩터 매우 낮음
        if composite < 40 or (risk < 25 and composite < 50):
            return "SELL"

        # Hold
        return "HOLD"

    def _generate_reasons(
        self, composite: float, fs: Dict, factor_data: Dict, tech: Dict
    ) -> List[str]:
        reasons = []

        # 모멘텀 분석
        mom = factor_data.get("momentum", {})
        rsi = mom.get("rsi")
        if rsi is not None:
            if rsi >= 70:
                reasons.append(f"RSI {rsi:.0f} — 과매수 구간, 단기 조정 가능성")
            elif rsi <= 30:
                reasons.append(f"RSI {rsi:.0f} — 과매도 구간, 반등 가능성")
            elif rsi >= 55:
                reasons.append(f"RSI {rsi:.0f} — 강세 모멘텀 유지")

        # 밸류에이션
        val = factor_data.get("value", {})
        fpe = val.get("forward_pe")
        if fpe is not None:
            if fpe < 15:
                reasons.append(f"Forward P/E {fpe:.1f} — 저평가 영역")
            elif fpe > 40:
                reasons.append(f"Forward P/E {fpe:.1f} — 고평가 주의")

        # 성장성
        grow = factor_data.get("growth", {})
        rev_g = grow.get("revenue_growth")
        if rev_g is not None:
            if rev_g > 20:
                reasons.append(f"매출 성장률 {rev_g:.1f}% — 고성장")
            elif rev_g < 0:
                reasons.append(f"매출 성장률 {rev_g:.1f}% — 역성장 우려")

        # 수급
        flow = factor_data.get("flow", {})
        vol_ratio = flow.get("volume_ratio")
        if vol_ratio is not None and vol_ratio > 2.0:
            reasons.append(f"거래량 {vol_ratio:.1f}배 급증 — 수급 이상 신호")

        # 팩터 강점/약점
        best_factor = max(fs, key=fs.get) if fs else None
        worst_factor = min(fs, key=fs.get) if fs else None
        if best_factor:
            reasons.append(f"최강 팩터: {best_factor} ({fs[best_factor]:.0f}점)")
        if worst_factor and worst_factor != best_factor:
            reasons.append(f"최약 팩터: {worst_factor} ({fs[worst_factor]:.0f}점)")

        return reasons[:6]  # 최대 6개

    def _check_alerts(
        self, factor_data: Dict, tech: Dict, raw_stock: Dict
    ) -> List[Dict]:
        alerts = []

        mom = factor_data.get("momentum", {})
        val = factor_data.get("value", {})
        grow = factor_data.get("growth", {})
        flow = factor_data.get("flow", {})

        # 모멘텀 반전 시그널
        rsi = mom.get("rsi")
        macd_hist = mom.get("macd_histogram")
        if rsi is not None and macd_hist is not None:
            if rsi <= 30 and macd_hist > 0:
                alerts.append({
                    "type": "MOMENTUM_REVERSAL",
                    "severity": "info",
                    "message": f"모멘텀 반전 후보 — RSI {rsi:.0f}(과매도) + MACD 양전환",
                })

        # 과열 경고
        bb_pct = mom.get("bb_pct_b")
        vol_ratio = flow.get("volume_ratio")
        if rsi is not None and rsi >= 75:
            if bb_pct is not None and bb_pct > 1.0:
                alerts.append({
                    "type": "OVERHEAT",
                    "severity": "warning",
                    "message": f"과열 경고 — RSI {rsi:.0f}, 볼린저밴드 상단 이탈",
                })

        # 가치 함정 (Value Trap)
        fpe = val.get("forward_pe")
        earn_g = grow.get("earnings_growth")
        if fpe is not None and earn_g is not None:
            if fpe < 12 and earn_g < -10:
                alerts.append({
                    "type": "VALUE_TRAP",
                    "severity": "danger",
                    "message": f"가치 함정 위험 — 저PER({fpe:.0f}) + 이익 역성장({earn_g:.0f}%)",
                })

        # 수급 이상
        if vol_ratio is not None and vol_ratio >= 3.0:
            alerts.append({
                "type": "VOLUME_SPIKE",
                "severity": "info",
                "message": f"거래량 급등 — 20일 평균 대비 {vol_ratio:.1f}배",
            })

        # 공매도 경고
        short_pct = flow.get("short_pct_float")
        if short_pct is not None and short_pct > 10:
            alerts.append({
                "type": "HIGH_SHORT",
                "severity": "warning",
                "message": f"공매도 비율 높음 — 유통주식 대비 {short_pct:.1f}%",
            })

        return alerts
