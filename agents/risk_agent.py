"""
risk_agent.py — RiskGuard 🛡️ 리스크 관리자
"""
from .base_agent import BaseAgent, AgentReport, AgentCritique

SYSTEM_PROMPT = """당신은 'RiskGuard(리스크 관리자)'라는 이름의 위험 관리 전문가입니다.

[페르소나]
- 모든 투자의 하방 리스크를 최우선으로 평가합니다.
- 변동성, 최대 손실(MDD), 베타, 샤프 비율로 위험-수익 효율을 판단합니다.
- "수익보다 원금 보전"이 최우선 원칙입니다.
- 다른 에이전트가 놓치는 리스크 요인을 반드시 지적합니다.

[분석 원칙]
- Beta > 1.5 = 고위험, < 0.8 = 방어적
- 연환산 변동성 30%+ = 고변동
- MDD -30% 이하 = 심각한 손실 이력
- Sharpe < 0.5 = 리스크 대비 수익 부족
- 공매도 비율 높음 = 하락 압력 경고

[출력] 한국어 / 반드시 JSON 형식으로만 응답"""


class RiskAgent(BaseAgent):
    def __init__(self, client, model):
        super().__init__(client, model)
        self.name = "RiskGuard"
        self.role = "리스크 관리자"
        self.avatar = "🛡️"
        self.system_prompt = SYSTEM_PROMPT

    def analyze(self, ticker: str, stock_data: dict, quant_data: dict) -> AgentReport:
        summary = self._stock_summary(stock_data, quant_data)
        prompt = f"""{summary}

위 종목의 리스크 프로파일을 분석하세요.

반드시 아래 JSON으로만 응답:
{{
  "analysis": "300자 이상 리스크 분석 (변동성, Beta, MDD, Sharpe 등 수치 필수)",
  "key_points": ["핵심1", "핵심2", "핵심3"],
  "confidence_score": 75,
  "stance": "NEUTRAL"
}}

stance: BULLISH(리스크 낮음) / NEUTRAL / BEARISH(리스크 높음)"""

        result = self._call_llm([{"role": "user", "content": prompt}])
        data = self._parse_json_response(result)
        return AgentReport(
            agent_name=self.name, role=self.role, avatar=self.avatar,
            ticker=ticker,
            analysis=data.get("analysis", result[:800]),
            key_points=data.get("key_points", ["분석 완료"]),
            confidence_score=max(0, min(100, int(data.get("confidence_score", 50)))),
            stance=data.get("stance", "NEUTRAL").upper(),
        )

    def critique(self, other_report: AgentReport, stock_data: dict) -> AgentCritique:
        prompt = f"""리스크 관리자로서 아래 분석에 반론을 제시하세요.

[{other_report.agent_name} — {other_report.role}의 분석]
의견: {other_report.stance} (확신도: {other_report.confidence_score})
주장: {other_report.analysis[:400]}

리스크 관점에서 간과한 위험 요인을 150~250자로 지적:"""
        result = self._call_llm([{"role": "user", "content": prompt}])
        return AgentCritique(from_agent=self.name, to_agent=other_report.agent_name, critique=result[:300])
