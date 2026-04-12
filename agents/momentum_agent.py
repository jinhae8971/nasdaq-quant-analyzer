"""
momentum_agent.py — MomentumBot 📊 추세/모멘텀 분석가
"""
from .base_agent import BaseAgent, AgentReport, AgentCritique

SYSTEM_PROMPT = """당신은 'MomentumBot(모멘텀 분석가)'이라는 이름의 차트 전문가입니다.

[페르소나]
- RSI, MACD, 볼린저밴드, 이동평균 등 기술적 지표로 단기~중기 방향성을 판단합니다.
- 추세의 강도와 지속 가능성을 정량적으로 평가합니다.
- 감이 아닌 숫자로만 판단합니다.

[분석 원칙]
- RSI 50+ = 강세 모멘텀, 70+ = 과매수, 30- = 과매도
- MACD 히스토그램 방향과 크로스 반드시 확인
- 이동평균선 배열(5/20/60/200일)로 추세 판단
- 52주 고점 대비 위치로 상승 여력 평가
- 볼린저밴드 포지션으로 변동성 국면 판단

[출력] 한국어 / 반드시 JSON 형식으로만 응답"""


class MomentumAgent(BaseAgent):
    def __init__(self, client, model):
        super().__init__(client, model)
        self.name = "MomentumBot"
        self.role = "모멘텀 분석가"
        self.avatar = "📊"
        self.system_prompt = SYSTEM_PROMPT

    def analyze(self, ticker: str, stock_data: dict, quant_data: dict) -> AgentReport:
        summary = self._stock_summary(stock_data, quant_data)
        prompt = f"""{summary}

위 종목의 기술적 모멘텀을 분석하세요.

반드시 아래 JSON으로만 응답:
{{
  "analysis": "300자 이상 기술적 분석 (RSI, MACD, 이동평균, BB 등 구체적 수치 필수)",
  "key_points": ["핵심1 (수치 포함)", "핵심2", "핵심3"],
  "confidence_score": 75,
  "stance": "BULLISH"
}}

stance: BULLISH / NEUTRAL / BEARISH 중 하나
confidence_score: 0~100"""

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
        prompt = f"""모멘텀 분석가로서 아래 분석에 반론을 제시하세요.

[{other_report.agent_name} — {other_report.role}의 분석]
의견: {other_report.stance} (확신도: {other_report.confidence_score})
주장: {other_report.analysis[:400]}

기술적 지표 기반으로 150~250자 반론:"""
        result = self._call_llm([{"role": "user", "content": prompt}])
        return AgentCritique(from_agent=self.name, to_agent=other_report.agent_name, critique=result[:300])
