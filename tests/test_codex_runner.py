"""codex_runner.py: factory patch wiring, kwarg filtering, rate-limit retry. No network."""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage
from langchain_openai.chatgpt_oauth import _ChatGPTToken

import codex_runner
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph import trading_graph
from tradingagents.llm_clients.openai_client import NormalizedChatOpenAI


class FakeTokenProvider:
    _token = _ChatGPTToken(access_token="a", refresh_token="r",
                           expires_at=datetime.now(UTC) + timedelta(hours=1), account_id="acct")

    def get_token(self):
        return self._token

    async def aget_token(self):
        return self._token

    def get_access_token(self):
        return "a"

    async def aget_access_token(self):
        return "a"


def _rate_limit_error(headers=None, message="rate limited"):
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
    response = httpx.Response(429, headers=headers or {}, request=request)
    return openai.RateLimitError(message, response=response, body=None)


def test_patch_routes_deep_and_quick_through_codex(tmp_path, monkeypatch):
    monkeypatch.setattr(trading_graph, "create_llm_client", trading_graph.create_llm_client)
    codex_runner.install_patch()
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(llm_provider="codex", quick_think_llm="quick-model", deep_think_llm="deep-model",
               temperature=0.3, max_tokens=100, llm_max_retries=4,
               results_dir=str(tmp_path / "r"), data_cache_dir=str(tmp_path / "c"),
               memory_log_path=str(tmp_path / "log.md"))

    graph = trading_graph.TradingAgentsGraph(selected_analysts=["market"], config=cfg)

    for llm, model in ((graph.quick_thinking_llm, "quick-model"),
                       (graph.deep_thinking_llm, "deep-model")):
        assert isinstance(llm, codex_runner.CodexChat)
        assert llm.model_name == model
        assert llm.temperature is None and llm.max_tokens is None
        assert llm.max_retries == 4


def test_payload_meets_codex_constraints():
    llm = codex_runner.CodexChat(model="m", token_provider=FakeTokenProvider(), temperature=None)
    payload = llm._get_request_payload([("system", "be terse"), ("user", "hi")])
    assert payload["stream"] is True and payload["store"] is False
    assert payload["instructions"] == "be terse"
    assert "temperature" not in payload
    assert all(item.get("role") != "system" for item in payload["input"])
    assert payload["extra_headers"]["ChatGPT-Account-Id"] == "acct"


@pytest.mark.parametrize(("headers", "message", "expected"), [
    ({"retry-after": "42"}, "rate limited", 42.0),
    ({}, "Usage limit reached. Try again in 1h2m.", 3720.0),
    ({}, "Please retry in 30s", 30.0),
    ({}, "rate limited", None),
])
def test_retry_after_parsing(headers, message, expected):
    assert codex_runner.retry_after_seconds(_rate_limit_error(headers, message)) == expected


def test_invoke_sleeps_through_rate_limit(monkeypatch):
    calls, sleeps = [], []

    def flaky_invoke(self, input, config=None, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise _rate_limit_error({"retry-after": "7"})
        return AIMessage(content="ok")

    monkeypatch.setattr(NormalizedChatOpenAI, "invoke", flaky_invoke)
    monkeypatch.setattr(codex_runner.time, "sleep", sleeps.append)
    llm = codex_runner.CodexChat(model="m", token_provider=FakeTokenProvider())

    assert llm.invoke("hi").content == "ok"
    assert sleeps == [12.0]  # retry-after + 5s buffer
