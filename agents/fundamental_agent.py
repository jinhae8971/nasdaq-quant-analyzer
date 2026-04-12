"""
fundamental_agent.py — ValueHunter 💎 펀더멘털 분석가
"""
from .base_agent import BaseAgent, AgentReport, AgentCritique

SYSTEM_PROMPT = """당신은 'ValueHunter(가치 분석가)'라는 이름의 펀더멘털 전문가입니다.

[페르소나]
- 밸류에이션(P/E, P/S, PEG, EV/EBITDA)과 재무 건전성으로 내재가치를 평가합니다.
- 실적 성장률, 마진 트렌드, ROE/ROA로 기업의 수익 창출 능력을 판단합니다.
- 단기 주가가 아닌 기업의 본질적 가치에 집중합니다.
- 워렌 버핏의 "적정 가격에 훌륭한 기업을 사라" 철학을 따릅니다.

[분석 원칙]
- Forward P/E가 섹터 평균 대비 어떤 위치인지 확인
- PEG < 1 = 성장 대비 저평가, PEG > 2 = 고평가
- ROE 15%+ = 우수, Debt/Equity 적정 여부
- FCF 양수 + 성장 = 이상적
- 애널리스트 목표가 대비 업사이드 잠재력

[출력] 한국어 / 반드시 JSON 형식으로만 응답"""


class FundamentalAgent(BaseAgent):
    def __init__(self, client, model):
        super().__init__(client, model)
        self.name = "ValueHunter"
        self.role = "펀더멘털 분석가"
        self.avatar = "💎"
        self.system_prompt = SYSTEM_PROMPT

    def analyze(self, ticker: str, stock_data: dict, quant_data: dict) -> AgentReport:
        summary = self._stock_summary(stock_data, quant_data)
        prompt = f"""{summary}

위 종목의 펀더멘털과 밸류에이션을 분석하세요.

반드시 아래 JSON으로만 응답:
{{
  "analysis": "300자 이상 펀더멘털 분석 (밸류에이션, 실적, 재무건전성 수치 필수)",
  "key_points": ["핵심1", "핵심2", "핵심3"],
  "confidence_score": 75,
  "stance": "BULLISH"
}}

stance: BULLISH / NEUTRAL / BEARISH 중 하나"""

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
        prompt = f"""펀더멘털 분석가로서 아래 분석에 반론을 제시하세요.

[{other_report.agent_name} — {other_report.role}의 분석]
의견: {other_report.stance} (확신도: {other_report.confidence_score})
주장: {other_report.analysis[:400]}

밸류에이션과 재무 지표 기반으로 150~250자 반론:"""
        result = self._call_llm([{"role": "user", "content": prompt}])
        return AgentCritique(from_agent=self.name, to_agent=other_report.agent_name, critique=result[:300])
