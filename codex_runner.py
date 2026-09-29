"""Run TradingAgents on a ChatGPT/Codex subscription instead of an API key.

Uses langchain-openai's experimental ``_ChatOpenAICodex`` (ChatGPT OAuth
against the Codex backend) and monkeypatches the LLM factory that
``TradingAgentsGraph`` calls, so no upstream package code is modified.

Usage::

    python codex_runner.py login              # device code (headless-friendly)
    python codex_runner.py login --browser    # loopback browser flow
    python codex_runner.py run NVDA 2026-09-01
    python codex_runner.py run NVDA 2026-09-01 --quick gpt-6-luna --deep gpt-6-sol \\
        --analysts market news

Tokens live in langchain's own store, ``~/.langchain/chatgpt-auth.json``
(never ``~/.codex/auth.json``). Model defaults come from
``DEFAULT_CONFIG`` (quick_think_llm / deep_think_llm); the Codex backend
must actually serve the IDs you pick, which is only checked on first call.
Use only where your OpenAI account/plan terms permit ChatGPT-authenticated
Codex access.
"""

from __future__ import annotations

import argparse
import logging
import re
import time

import openai

# ponytail: private, experimental class (langchain-openai 1.6.6). Pin
# langchain-openai if this import or its forced-field contract changes.
from langchain_openai.chat_models.codex import _ChatOpenAICodex

import tradingagents.graph.trading_graph as trading_graph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.openai_client import NormalizedChatOpenAI

logger = logging.getLogger(__name__)

RATE_LIMIT_MAX_WAITS = 12
RATE_LIMIT_MAX_WAIT_SECONDS = 3600
_ESCALATING_WAITS = (60, 120, 300, 600, 900)
_DURATION_RE = re.compile(
    r"(?:try again|retry|resets?)[^0-9]*"
    r"(?:(\d+)\s*h(?:ours?)?)?\s*(?:(\d+)\s*m(?:in(?:utes?)?)?)?\s*"
    r"(?:([\d.]+)\s*s(?:ec(?:onds?)?)?)?",
    re.IGNORECASE,
)
# Codex rejects temperature and max_(output_)tokens with a 400, so only these
# kwargs from build_llm_kwargs() are forwarded.
_PASSTHROUGH_KWARGS = ("callbacks", "max_retries", "reasoning_effort")


def retry_after_seconds(exc: BaseException) -> float | None:
    """Reset delay from a rate-limit error's headers or message, if present."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for key in ("retry-after", "x-ratelimit-reset-after", "x-ratelimit-reset-requests"):
        try:
            return float(headers[key])
        except (KeyError, TypeError, ValueError):
            continue
    match = _DURATION_RE.search(str(exc))
    if match and any(match.groups()):
        h, m, s = (float(g or 0) for g in match.groups())
        return h * 3600 + m * 60 + s or None
    return None


class CodexChat(NormalizedChatOpenAI, _ChatOpenAICodex):
    """Codex model with TradingAgents' content normalization, capability-driven
    structured output, and a sleep-through-quota retry around invoke.

    bind_tools / with_structured_output bindings delegate to this invoke too.
    """

    def invoke(self, input, config=None, **kwargs):
        for attempt in range(1, RATE_LIMIT_MAX_WAITS + 1):
            try:
                return super().invoke(input, config, **kwargs)
            except openai.RateLimitError as exc:
                if attempt == RATE_LIMIT_MAX_WAITS:
                    raise
                wait = retry_after_seconds(exc)
                if wait is None:
                    wait = _ESCALATING_WAITS[min(attempt, len(_ESCALATING_WAITS)) - 1]
                wait = min(wait + 5, RATE_LIMIT_MAX_WAIT_SECONDS)
                logger.warning("codex rate limited (%s); sleeping %.0fs (%d/%d)",
                               exc, wait, attempt, RATE_LIMIT_MAX_WAITS)
                time.sleep(wait)


class _CodexClient:
    def __init__(self, llm):
        self._llm = llm

    def get_llm(self):
        return self._llm


def make_codex_llm(model: str, **kwargs) -> CodexChat:
    kept = {k: v for k, v in kwargs.items() if k in _PASSTHROUGH_KWARGS}
    return CodexChat(model=model, originator="tradingagents", **kept)


def codex_create_llm_client(provider, model, base_url=None, **kwargs):
    """Drop-in for create_llm_client: provider/base_url are ignored."""
    return _CodexClient(make_codex_llm(model, **kwargs))


def install_patch() -> None:
    # ponytail: patches the name trading_graph imported, not the factory module.
    trading_graph.create_llm_client = codex_create_llm_client


def cmd_login(args) -> None:
    from langchain_openai.chatgpt_oauth import login_chatgpt, login_chatgpt_device

    provider = login_chatgpt() if args.browser else login_chatgpt_device()
    print(f"Logged in; token stored at {provider.path}")


def cmd_run(args) -> None:
    install_patch()
    config = DEFAULT_CONFIG.copy()
    # Not a registered provider; only recorded in reports (build_llm_kwargs then
    # skips openai_reasoning_effort, which Codex would accept anyway).
    config["llm_provider"] = "codex"
    config["quick_think_llm"] = args.quick
    config["deep_think_llm"] = args.deep
    graph = trading_graph.TradingAgentsGraph(
        selected_analysts=args.analysts, debug=args.debug, config=config
    )
    _, decision = graph.propagate(args.ticker, args.date)
    print(decision)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    login = sub.add_parser("login", help="Sign in with ChatGPT (device code by default)")
    login.add_argument("--browser", action="store_true", help="Use the loopback browser flow")
    login.set_defaults(func=cmd_login)

    run = sub.add_parser("run", help="Run TradingAgents for TICKER on DATE (YYYY-MM-DD)")
    run.add_argument("ticker")
    run.add_argument("date")
    run.add_argument("--quick", default=DEFAULT_CONFIG["quick_think_llm"])
    run.add_argument("--deep", default=DEFAULT_CONFIG["deep_think_llm"])
    run.add_argument("--analysts", nargs="+",
                     default=["market", "social", "news", "fundamentals"])
    run.add_argument("--debug", action="store_true")
    run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
