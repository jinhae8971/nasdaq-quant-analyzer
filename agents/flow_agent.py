"""
flow_agent.py — FlowTracker 🌊 수급 분석가
"""
from .base_agent import BaseAgent, AgentReport, AgentCritique

SYSTEM_PROMPT = """당신은 'FlowTracker(수급 분석가)'라는 이름의 자금흐름 전문가입니다.

[페르소나]
- 거래량 패턴, 기관 매매 동향, 공매도 비율로 수급 압력을 판단합니다.
- "가격은 수급이 만든다"는 원칙을 따릅니다.
- 이상 거래량을 포착하여 스마트머니의 움직임을 추적합니다.
- 내부자 거래 신호를 주시합니다.

[분석 원칙]
- 거래량 2배+ 급증 = 수급 변화 신호
- 기관 보유율 60%+ = 안정적 수급
- 공매도 비율 10%+ = 하락 압력
- 내부자 순매수 = 긍정적 신호
- 거래량 감소 + 주가 상승 = 탈진 가능성

[출력] 한국어 / 반드시 JSON 형식으로만 응답"""


class FlowAgent(BaseAgent):
    def __init__(self, client, model):
        super().__init__(client, model)
        self.name = "FlowTracker"
        self.role = "수급 분석가"
        self.avatar = "🌊"
        self.system_prompt = SYSTEM_PROMPT

    def analyze(self, ticker: str, stock_data: dict, quant_data: dict) -> AgentReport:
        summary = self._stock_summary(stock_data, quant_data)
        prompt = f"""{summary}

위 종목의 수급 상황과 자금흐름을 분석하세요.

반드시 아래 JSON으로만 응답:
{{
  "analysis": "300자 이상 수급 분석 (거래량, 기관, 공매도 등 수치 필수)",
  "key_points": ["핵심1", "핵심2", "핵심3"],
  "confidence_score": 75,
  "stance": "BULLISH"
}}

stance: BULLISH(수급 유입) / NEUTRAL / BEARISH(수급 이탈)"""

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
        prompt = f"""수급 분석가로서 아래 분석에 반론을 제시하세요.

[{other_report.agent_name} — {other_report.role}의 분석]
의견: {other_report.stance} (확신도: {other_report.confidence_score})
주장: {other_report.analysis[:400]}

수급/자금흐름 관점에서 150~250자 반론:"""
        result = self._call_llm([{"role": "user", "content": prompt}])
        return AgentCritique(from_agent=self.name, to_agent=other_report.agent_name, critique=result[:300])
