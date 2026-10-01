"""LLM providers behind one small interface, so the same agent loop runs on any of them.

    AnthropicProvider    - Claude via the Anthropic Messages API (with prompt caching)
    GeminiProvider       - Gemini via Google's own google-genai SDK (free tier; works with the
                           new "AQ." API keys, which Google's OpenAI-compatible endpoint rejects)
    OpenAICompatProvider - any OpenAI-compatible chat API (OpenAI, ...)

Each provider keeps the conversation in its own wire format and exposes:
    step()                -> Turn(texts, tool_calls, input_tokens, output_tokens, stop_reason)
    add_tool_results(...) -> feed tool outputs back for the next step
"""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field

import truststore

# Verify TLS against the OS certificate store rather than the bundled certifi list, so corporate/
# antivirus HTTPS-inspection proxies that install their root CA into the OS trust store (common on
# Windows) are trusted here the same way the system and browsers already trust them.
truststore.inject_into_ssl()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Turn:
    texts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, client=None, max_tokens: int = 4096):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.model, self.max_tokens = client, model, max_tokens

    def start(self, system: str, tools: list[dict], user: str) -> None:
        # Prompt caching: system prompt, tool definitions and the growing conversation are
        # re-sent every step, so cache breakpoints make the repeated prefix much cheaper.
        self.system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        self.tools = tools[:-1] + [dict(tools[-1], cache_control={"type": "ephemeral"})]
        self.messages: list[dict] = [{"role": "user", "content": user}]

    def step(self) -> Turn:
        resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                           system=self.system, tools=self.tools, messages=self.messages)
        self.messages.append({"role": "assistant", "content": resp.content})
        turn = Turn(input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
                    stop_reason=resp.stop_reason or "")
        for b in resp.content:
            if b.type == "text":
                turn.texts.append(b.text)
            elif b.type == "tool_use":
                turn.tool_calls.append(ToolCall(b.id, b.name, dict(b.input)))
        return turn

    def add_tool_results(self, results: list[tuple[ToolCall, str]]) -> None:
        for m in self.messages:  # keep a single rolling cache breakpoint on the newest turn
            if m["role"] == "user" and isinstance(m["content"], list):
                for c in m["content"]:
                    if isinstance(c, dict):
                        c.pop("cache_control", None)
        content = [{"type": "tool_result", "tool_use_id": call.id, "content": out} for call, out in results]
        content[-1]["cache_control"] = {"type": "ephemeral"}
        self.messages.append({"role": "user", "content": content})


def _clean_schema(schema: dict) -> dict:
    """Drop JSON-schema keys some OpenAI-compatible backends (e.g. Gemini) reject."""
    s = copy.deepcopy(schema)

    def walk(node):
        if isinstance(node, dict):
            node.pop("default", None)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(s)
    return s


def to_openai_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        fn = {"name": t["name"], "description": t["description"]}
        params = _clean_schema(t["input_schema"])
        if params.get("properties"):  # parameterless tools omit the schema entirely
            fn["parameters"] = params
        out.append({"type": "function", "function": fn})
    return out


class GeminiProvider:
    """Gemini through the native google-genai SDK."""
    name = "gemini"

    def __init__(self, model: str, api_key: str | None = None, client=None, rpm: float | None = None,
                 max_tokens: int = 4096, max_retries: int = 8):
        from google.genai import types
        self.types = types
        if client is None:
            from google import genai
            client = genai.Client(api_key=api_key)
        self.client, self.model, self.max_tokens, self.max_retries = client, model, max_tokens, max_retries
        self.min_interval = 60.0 / rpm if rpm else 0.0
        self._last_call = 0.0

    def start(self, system: str, tools: list[dict], user: str) -> None:
        t = self.types
        decls = []
        for tool in tools:
            params = _clean_schema(tool["input_schema"])
            kw = {"parameters_json_schema": params} if params.get("properties") else {}
            decls.append(t.FunctionDeclaration(name=tool["name"], description=tool["description"], **kw))
        self.config = t.GenerateContentConfig(
            system_instruction=system, max_output_tokens=self.max_tokens,
            tools=[t.Tool(function_declarations=decls)],
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True))
        self.contents = [t.Content(role="user", parts=[t.Part(text=user)])]

    def _generate(self):
        from google.genai import errors
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                return self.client.models.generate_content(model=self.model, contents=self.contents,
                                                           config=self.config)
            except errors.APIError as e:  # free tier: back off on rate limits / overload
                if getattr(e, "code", None) in (429, 500, 503) and attempt < self.max_retries:
                    time.sleep(min(60, 5 * 2 ** attempt))
                    continue
                raise

    def step(self) -> Turn:
        resp = self._generate()
        turn = Turn()
        if resp.usage_metadata:
            u = resp.usage_metadata
            turn.input_tokens = u.prompt_token_count or 0
            turn.output_tokens = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
        cand = resp.candidates[0] if resp.candidates else None
        if cand is None or cand.content is None:
            turn.stop_reason = str(getattr(cand, "finish_reason", "") or "empty_response")
            return turn
        turn.stop_reason = str(cand.finish_reason or "")
        # Append the model's content unchanged: it carries thought signatures Gemini needs back.
        self.contents.append(cand.content)
        for i, part in enumerate(cand.content.parts or []):
            if part.text and not getattr(part, "thought", False):
                turn.texts.append(part.text)
            if part.function_call:
                fc = part.function_call
                turn.tool_calls.append(ToolCall(fc.id or f"{fc.name}_{len(self.contents)}_{i}",
                                                fc.name, dict(fc.args or {})))
        return turn

    def add_tool_results(self, results: list[tuple[ToolCall, str]]) -> None:
        t = self.types
        parts = []
        for call, out in results:
            try:
                payload = json.loads(out)
            except json.JSONDecodeError:
                payload = out
            parts.append(t.Part(function_response=t.FunctionResponse(
                id=None if call.id.startswith(f"{call.name}_") else call.id,
                name=call.name, response={"result": payload})))
        self.contents.append(t.Content(role="user", parts=parts))


class OpenAICompatProvider:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 client=None, rpm: float | None = None, name: str = "openai", max_tokens: int = 4096):
        if client is None:
            import openai
            client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=8)
        self.client, self.model, self.name, self.max_tokens = client, model, name, max_tokens
        self.min_interval = 60.0 / rpm if rpm else 0.0  # stay under free-tier requests/minute
        self._last_call = 0.0

    def start(self, system: str, tools: list[dict], user: str) -> None:
        self.tools = to_openai_tools(tools)
        self.messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def step(self) -> Turn:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()
        resp = self.client.chat.completions.create(model=self.model, messages=self.messages,
                                                   tools=self.tools, max_tokens=self.max_tokens)
        choice = resp.choices[0]
        msg = choice.message
        turn = Turn(stop_reason=choice.finish_reason or "")
        if resp.usage:
            turn.input_tokens = resp.usage.prompt_tokens or 0
            turn.output_tokens = resp.usage.completion_tokens or 0
        if msg.content:
            turn.texts.append(msg.content)
        assistant = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant["tool_calls"] = []
            for i, tc in enumerate(msg.tool_calls):
                call_id = tc.id or f"call_{len(self.messages)}_{i}"
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_unparseable_arguments": tc.function.arguments}
                turn.tool_calls.append(ToolCall(call_id, tc.function.name, args))
                assistant["tool_calls"].append({"id": call_id, "type": "function", "function": {
                    "name": tc.function.name, "arguments": tc.function.arguments or "{}"}})
        self.messages.append(assistant)
        return turn

    def add_tool_results(self, results: list[tuple[ToolCall, str]]) -> None:
        for call, out in results:
            self.messages.append({"role": "tool", "tool_call_id": call.id, "content": out})


def make_provider(provider: str, model: str | None, client=None, rpm: float | None = None):
    """Build a provider from a name. Keys come from the environment, never from arguments or files."""
    if provider == "anthropic":
        return AnthropicProvider(model or os.environ.get("DRCAGENT_MODEL", "claude-sonnet-5"), client=client)
    if provider == "gemini":
        model = model or os.environ.get("DRCAGENT_GEMINI_MODEL")
        if not model:
            raise SystemExit("Choose a Gemini model with --model (list them: python -m drcagent models --provider gemini)")
        key = os.environ.get("GEMINI_API_KEY")
        if client is None and not key:
            raise SystemExit("Set GEMINI_API_KEY first (free key from aistudio.google.com)")
        return GeminiProvider(model, key, client=client, rpm=rpm or 9)
    if provider == "openai":
        return OpenAICompatProvider(model or "gpt-5-mini", client=client, rpm=rpm, name="openai")
    raise ValueError(f"unknown provider {provider}")


def list_models(provider: str) -> list[str]:
    if provider == "gemini":
        from google import genai
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        return sorted(m.name.removeprefix("models/") for m in client.models.list()
                      if not m.supported_actions or "generateContent" in m.supported_actions)
    if provider == "anthropic":
        import anthropic
        return [m.id for m in anthropic.Anthropic().models.list()]
    else:
        import openai
        client = openai.OpenAI()
    return sorted(m.id for m in client.models.list())
