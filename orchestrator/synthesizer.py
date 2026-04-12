"""
synthesizer.py — AI 종합 판단 엔진

4 에이전트 독립 분석 → 교차 반론 → 최종 종합 인사이트 생성
"""
import logging
from typing import Dict, List

from agents.base_agent import BaseAgent, AgentReport, AgentCritique

logger = logging.getLogger(__name__)

# 교차 반론 쌍: Momentum ↔ Value, Risk ↔ Flow
CRITIQUE_PAIRS = [
    (0, 1),  # Momentum → Value
    (1, 0),  # Value → Momentum
    (2, 3),  # Risk → Flow
    (3, 2),  # Flow → Risk
]


class Synthesizer:
    """4 에이전트 분석 + 교차 반론 + 종합 판단"""

    def __init__(self, agents: List[BaseAgent], client, model: str):
        self.agents = agents
        self.client = client
        self.model = model

    def analyze_stock(self, ticker: str, stock_data: dict, quant_data: dict) -> Dict:
        """단일 종목에 대한 멀티에이전트 심층분석"""

        # Phase 1: 독립 분석
        logger.info(f"  [{ticker}] Phase 1: 독립 분석")
        reports: List[AgentReport] = []
        for agent in self.agents:
            try:
                report = agent.analyze(ticker, stock_data, quant_data)
                reports.append(report)
                logger.info(f"    [{agent.name}] {report.stance} ({report.confidence_score}%)")
            except Exception as e:
                logger.error(f"    [{agent.name}] 분석 실패: {e}")
                reports.append(AgentReport(
                    agent_name=agent.name, role=agent.role, avatar=agent.avatar,
                    ticker=ticker, analysis=f"분석 오류: {str(e)[:100]}",
                    key_points=["분석 불가"], confidence_score=0, stance="NEUTRAL",
                ))

        # Phase 2: 교차 반론
        logger.info(f"  [{ticker}] Phase 2: 교차 반론")
        critiques: List[AgentCritique] = []
        for from_idx, to_idx in CRITIQUE_PAIRS:
            if from_idx >= len(self.agents) or to_idx >= len(reports):
                continue
            try:
                critique = self.agents[from_idx].critique(reports[to_idx], stock_data)
                critiques.append(critique)
            except Exception as e:
                logger.error(f"    반론 실패: {e}")

        # Phase 3: 종합 판단
        logger.info(f"  [{ticker}] Phase 3: AI 종합 판단")
        synthesis = self._synthesize(ticker, stock_data, quant_data, reports, critiques)

        return {
            "ticker":     ticker,
            "reports":    [r.to_dict() for r in reports],
            "critiques":  [c.to_dict() for c in critiques],
            "synthesis":  synthesis,
        }

    def _synthesize(
        self, ticker: str, stock: dict, quant: dict,
        reports: List[AgentReport], critiques: List[AgentCritique]
    ) -> Dict:
        # 에이전트 합의 계산
        stances = [r.stance for r in reports]
        bullish = sum(1 for s in stances if s == "BULLISH")
        bearish = sum(1 for s in stances if s == "BEARISH")
        avg_confidence = sum(r.confidence_score for r in reports) / max(len(reports), 1)

        # 합의 방향
        if bullish >= 3:
            consensus = "BULLISH"
        elif bearish >= 3:
            consensus = "BEARISH"
        elif bullish > bearish:
            consensus = "LEAN_BULLISH"
        elif bearish > bullish:
            consensus = "LEAN_BEARISH"
        else:
            consensus = "MIXED"

        # AI 종합 요약 생성
        reports_text = "\n".join([
            f"[{r.agent_name}] {r.stance}({r.confidence_score}%): {r.analysis[:200]}"
            for r in reports
        ])
        critiques_text = "\n".join([
            f"[{c.from_agent}→{c.to_agent}]: {c.critique[:150]}"
            for c in critiques
        ])

        prompt = f"""[{ticker}] {stock.get('name', '')} 종합 투자 인사이트를 작성하세요.

퀀트 스코어: {quant.get('composite_score', 'N/A')} ({quant.get('grade', '?')})
시그널: {quant.get('signal', 'N/A')}

에이전트 분석:
{reports_text}

교차 반론:
{critiques_text}

에이전트 합의: {consensus} (평균 확신도 {avg_confidence:.0f}%)

반드시 아래 JSON으로만 응답:
{{
  "summary": "3~5줄 핵심 투자 인사이트 (한국어)",
  "action": "매수/관망/매도 중 하나와 구체적 이유",
  "key_risk": "가장 주의할 리스크 1가지",
  "catalyst": "향후 주가 촉매 요인 1가지"
}}"""

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            import json, re
            text = resp.content[0].text
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                result = json.loads(match.group())
            else:
                result = {"summary": text[:500]}
        except Exception as e:
            logger.error(f"  [{ticker}] 종합 판단 실패: {e}")
            result = {"summary": f"종합 판단 생성 실패: {str(e)[:100]}"}

        result["consensus"] = consensus
        result["avg_confidence"] = round(avg_confidence, 1)
        result["stance_distribution"] = {
            "bullish": bullish,
            "neutral": len(stances) - bullish - bearish,
            "bearish": bearish,
        }
        return result
