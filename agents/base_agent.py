"""
base_agent.py — AI 에이전트 공통 기반 클래스
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class AgentReport:
    agent_name: str
    role: str
    avatar: str
    ticker: str
    analysis: str
    key_points: List[str]
    confidence_score: int   # 0~100
    stance: str             # BULLISH / NEUTRAL / BEARISH

    def to_dict(self) -> dict:
        return {
            "agent_name":       self.agent_name,
            "role":             self.role,
            "avatar":           self.avatar,
            "ticker":           self.ticker,
            "analysis":         self.analysis,
            "key_points":       self.key_points,
            "confidence_score": self.confidence_score,
            "stance":           self.stance,
        }


@dataclass
class AgentCritique:
    from_agent: str
    to_agent: str
    critique: str

    def to_dict(self) -> dict:
        return {
            "from_agent": self.from_agent,
            "to_agent":   self.to_agent,
            "critique":   self.critique,
        }


class BaseAgent:
    name: str = ""
    role: str = ""
    avatar: str = ""
    system_prompt: str = ""

    def __init__(self, client: anthropic.Anthropic, model: str):
        self.client = client
        self.model = model

    def analyze(self, ticker: str, stock_data: dict, quant_data: dict) -> AgentReport:
        raise NotImplementedError

    def critique(self, other_report: AgentReport, stock_data: dict) -> AgentCritique:
        raise NotImplementedError

    def _call_llm(self, messages: list) -> str:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                system=self.system_prompt,
                messages=messages,
            )
            return resp.content[0].text
        except Exception as e:
            logger.error(f"[{self.name}] LLM 호출 실패: {e}")
            return "{}"

    def _parse_json_response(self, text: str) -> dict:
        try:
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass
        return {}

    def _stock_summary(self, stock: dict, quant: dict) -> str:
        tech = stock.get("technical", {})
        fs = quant.get("factor_scores", {})
        return f"""[{stock.get('ticker', '?')}] {stock.get('name', '?')} | {stock.get('sector', '?')}
가격: ${stock.get('price', 'N/A')} ({stock.get('change_pct', 0):+.2f}%)
시가총액: ${stock.get('market_cap', 0):,.0f}
퀀트 스코어: {quant.get('composite_score', 'N/A')} ({quant.get('grade', '?')})
팩터 스코어: 모멘텀={fs.get('momentum', 'N/A')}, 가치={fs.get('value', 'N/A')}, 품질={fs.get('quality', 'N/A')}, 성장={fs.get('growth', 'N/A')}, 리스크={fs.get('risk', 'N/A')}, 수급={fs.get('flow', 'N/A')}
RSI: {tech.get('rsi', 'N/A')} | MACD: {tech.get('macd', 'N/A')} | BB%B: {tech.get('pct_b', 'N/A')}
Forward P/E: {stock.get('forward_pe', 'N/A')} | P/S: {stock.get('ps_ratio', 'N/A')} | PEG: {stock.get('peg_ratio', 'N/A')}
ROE: {stock.get('roe', 'N/A')} | Revenue Growth: {stock.get('revenue_growth', 'N/A')}
1M 수익률: {tech.get('return_1m', 'N/A')}% | 3M: {tech.get('return_3m', 'N/A')}% | 52주 고점 대비: {tech.get('from_52w_high_pct', 'N/A')}%
거래량 비율(20일): {stock.get('volume_ratio', 'N/A')}x | 기관 보유: {stock.get('held_pct_institutions', 'N/A')}"""
