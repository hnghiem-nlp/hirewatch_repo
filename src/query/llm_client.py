"""LLM API client with multi-provider support (OpenAI, Anthropic, Hyperbolic, OpenRouter)."""

import asyncio
import re
import time

from utils.config import load_settings, load_secrets
from utils.helpers import get_logger

logger = get_logger(__name__)

# Stores thinking content from the last query (if any).
# Access via llm_client.last_thinking after a query_llm() call.
last_thinking = None

# Stores token usage from the last query.
# Access via llm_client.last_usage after a query_llm() call.
last_usage = None

_THINK_PATTERN = re.compile(r"^(?:<think>)?(.*?)</think>", re.DOTALL)


def strip_thinking(text):
    """Strip thinking blocks, returning (clean_text, thinking_text).

    Handles both '<think>...</think>' and responses that start with thinking
    content followed by '</think>' (no opening tag).
    """
    if text is None:
        return "", None
    match = _THINK_PATTERN.search(text)
    if not match:
        return text, None
    thinking = match.group(1).strip()
    clean = text[match.end():].strip()
    return clean, thinking

PROVIDER_PREFIXES = {
    "openai": ["gpt-", "o1-", "o3-", "o4-"],
    "anthropic": ["claude-"],
    "openrouter": ["openai/", "anthropic/", "google/", "meta-llama/",
                    "deepseek/", "mistralai/", "qwen/", "x-ai/"],
    "hyperbolic": ["Qwen/"],  # capital Q = Hyperbolic's naming convention
}


def infer_provider(model: str) -> str:
    """Infer provider from model name."""
    for provider, prefixes in PROVIDER_PREFIXES.items():
        if any(model.startswith(p) for p in prefixes):
            return provider
    raise ValueError(
        f"Cannot infer provider for model '{model}'. "
        f"Use --provider to specify explicitly."
    )


def get_openai_client():
    """Create an OpenAI client."""
    from openai import OpenAI
    secrets = load_secrets()
    return OpenAI(api_key=secrets["OPENAI_API_KEY"])


def get_anthropic_client():
    """Create an Anthropic client."""
    from anthropic import Anthropic
    secrets = load_secrets()
    return Anthropic(api_key=secrets["ANTHROPIC_API_KEY"])


def get_hyperbolic_client():
    """Create an OpenAI-compatible client for Hyperbolic."""
    from openai import OpenAI
    secrets = load_secrets()
    return OpenAI(
        api_key=secrets["HYPERBOLIC_API_KEY"],
        base_url="https://api.hyperbolic.xyz/v1",
    )


def get_openrouter_client():
    """Create an OpenAI-compatible client for OpenRouter."""
    from openai import OpenAI
    secrets = load_secrets()
    return OpenAI(
        api_key=secrets["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )


def query_openai_compat(client, prompt, model, temperature, max_tokens, retry_attempts, retry_delay, extra_body=None):
    """Query any OpenAI-compatible API."""
    for attempt in range(retry_attempts):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if extra_body:
                kwargs["extra_body"] = extra_body
            response = client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            text = msg.content
            # Some OpenRouter-exposed reasoning models return the chain-of-thought
            # in a dedicated `reasoning` field alongside the answer in `content`.
            reasoning_text = getattr(msg, "reasoning", None)
            if reasoning_text:
                global last_thinking
                last_thinking = reasoning_text
            global last_usage
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            # Reasoning-capable models report the split as
            # completion_tokens_details.reasoning_tokens (OpenRouter / OpenAI).
            details = getattr(response.usage, "completion_tokens_details", None)
            if details is not None:
                rt = getattr(details, "reasoning_tokens", None)
                if rt is not None:
                    usage["reasoning_tokens"] = rt
            last_usage = usage
            logger.info("Response (model=%s, tokens=%d)",
                        model, response.usage.completion_tokens)
            return text
        except Exception as e:
            logger.warning("Attempt %d failed: %s", attempt + 1, e)
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
            else:
                raise


def query_openai(prompt, model, temperature, max_tokens, retry_attempts, retry_delay):
    """Query OpenAI API."""
    client = get_openai_client()
    # Newer models (gpt-5.x, o-series) require max_completion_tokens instead of max_tokens
    use_new_param = any(model.startswith(p) for p in ["gpt-5", "o1-", "o3-", "o4-"])
    for attempt in range(retry_attempts):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            if use_new_param:
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content
            global last_usage
            last_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            logger.info("OpenAI response (model=%s, tokens=%d)",
                        model, response.usage.completion_tokens)
            return text
        except Exception as e:
            logger.warning("Attempt %d failed: %s", attempt + 1, e)
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
            else:
                raise


def query_anthropic(prompt, model, temperature, max_tokens, retry_attempts, retry_delay):
    """Query Anthropic API."""
    client = get_anthropic_client()
    # Opus 4.7 and later models deprecate temperature (extended thinking by default)
    kwargs = {"model": model, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]}
    if not model.startswith("claude-opus-4-7"):
        kwargs["temperature"] = temperature
    for attempt in range(retry_attempts):
        try:
            response = client.messages.create(**kwargs)
            text = response.content[0].text
            global last_usage
            last_usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }
            logger.info("Anthropic response (model=%s, tokens=%d)",
                        model, response.usage.output_tokens)
            return text
        except Exception as e:
            logger.warning("Attempt %d failed: %s", attempt + 1, e)
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
            else:
                raise


def query_llm(prompt, model=None, temperature=None, max_tokens=None, provider=None,
              no_thinking=False, reasoning_effort=None):
    """Send a prompt to the LLM and return the response text.

    Args:
        prompt: The full prompt string.
        model: Model name (default from settings.json).
        temperature: Sampling temperature (default from settings.json).
        max_tokens: Max response tokens (default from settings.json).
        provider: "openai" or "anthropic" (auto-inferred from model name if not set).
        no_thinking: Disable thinking (legacy flag, equivalent to reasoning_effort="none").
        reasoning_effort: Control thinking budget: "none", "low", "medium", "high".
            For OpenRouter: passed as {"reasoning": {"effort": value}}.
            For Hyperbolic: "none" maps to {"enable_thinking": False}.

    Returns:
        Raw response text from the model.
    """
    settings = load_settings()
    llm_config = settings["llm"]

    model = model or llm_config["default_model"]
    temperature = temperature if temperature is not None else llm_config["temperature"]
    max_tokens = max_tokens or llm_config["max_tokens"]
    retry_attempts = llm_config["retry_attempts"]
    retry_delay = llm_config["retry_delay_seconds"]

    # Apply per-model overrides from settings.json
    overrides = llm_config.get("model_overrides", {}).get(model, {})
    if overrides.get("max_tokens"):
        max_tokens = max(max_tokens, overrides["max_tokens"])
    if overrides.get("no_thinking"):
        no_thinking = True
    if overrides.get("reasoning_effort"):
        reasoning_effort = overrides["reasoning_effort"]
    reasoning_max_tokens = overrides.get("reasoning_max_tokens")

    # Normalize: no_thinking is shorthand for reasoning_effort="none"
    if no_thinking and not reasoning_effort:
        reasoning_effort = "none"

    provider = provider or infer_provider(model)

    # Thinking models need higher token budget to fit reasoning + answer
    if "Thinking" in model or "thinking" in model:
        if reasoning_effort != "none":
            max_tokens = max(max_tokens, 16384)

    global last_thinking, last_usage
    last_thinking = None
    last_usage = None

    # Build extra_body for thinking control
    extra_body = None
    if reasoning_max_tokens is not None:
        # Explicit token budget overrides effort (for models like Gemini Pro where effort=none is rejected)
        extra_body = {"reasoning": {"max_tokens": reasoning_max_tokens}}
    elif reasoning_effort:
        if provider == "hyperbolic":
            extra_body = {"enable_thinking": False} if reasoning_effort == "none" else None
        elif provider == "openrouter":
            extra_body = {"reasoning": {"effort": reasoning_effort}}

    if provider == "openai":
        raw = query_openai(prompt, model, temperature, max_tokens, retry_attempts, retry_delay)
    elif provider == "anthropic":
        raw = query_anthropic(prompt, model, temperature, max_tokens, retry_attempts, retry_delay)
    elif provider == "hyperbolic":
        client = get_hyperbolic_client()
        raw = query_openai_compat(client, prompt, model, temperature, max_tokens, retry_attempts, retry_delay, extra_body=extra_body)
    elif provider == "openrouter":
        client = get_openrouter_client()
        raw = query_openai_compat(client, prompt, model, temperature, max_tokens, retry_attempts, retry_delay, extra_body=extra_body)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    clean, thinking = strip_thinking(raw)
    if thinking:
        last_thinking = thinking
        logger.info("Stripped %d chars of thinking content", len(thinking))
    return clean


# --- Async support ---

def get_async_openai_client():
    from openai import AsyncOpenAI
    secrets = load_secrets()
    return AsyncOpenAI(api_key=secrets["OPENAI_API_KEY"])


def get_async_anthropic_client():
    from anthropic import AsyncAnthropic
    secrets = load_secrets()
    return AsyncAnthropic(api_key=secrets["ANTHROPIC_API_KEY"])


def get_async_hyperbolic_client():
    from openai import AsyncOpenAI
    secrets = load_secrets()
    return AsyncOpenAI(
        api_key=secrets["HYPERBOLIC_API_KEY"],
        base_url="https://api.hyperbolic.xyz/v1",
    )


def get_async_openrouter_client():
    from openai import AsyncOpenAI
    secrets = load_secrets()
    return AsyncOpenAI(
        api_key=secrets["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )


async def async_query_openai_compat(client, prompt, model, temperature, max_tokens,
                                     retry_attempts, retry_delay, extra_body=None):
    """Async query for any OpenAI-compatible API."""
    # Newer OpenAI models require max_completion_tokens
    use_new_param = any(model.startswith(p) for p in ["gpt-5", "o1-", "o3-", "o4-"])
    for attempt in range(retry_attempts):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            if use_new_param:
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens
            if extra_body:
                kwargs["extra_body"] = extra_body
            response = await client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content
            logger.info("Async response (model=%s, tokens=%d)",
                        model, response.usage.completion_tokens)
            return text
        except Exception as e:
            logger.warning("Async attempt %d failed: %s", attempt + 1, e)
            if attempt < retry_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                raise


async def async_query_anthropic(client, prompt, model, temperature, max_tokens,
                                 retry_attempts, retry_delay):
    """Async query for Anthropic API."""
    # Opus 4.7 and later models deprecate temperature (extended thinking by default)
    kwargs = {"model": model, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]}
    if not model.startswith("claude-opus-4-7"):
        kwargs["temperature"] = temperature
    for attempt in range(retry_attempts):
        try:
            response = await client.messages.create(**kwargs)
            text = response.content[0].text
            logger.info("Async Anthropic response (model=%s, tokens=%d)",
                        model, response.usage.output_tokens)
            return text
        except Exception as e:
            logger.warning("Async attempt %d failed: %s", attempt + 1, e)
            if attempt < retry_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                raise


async def async_query_llm(prompt, model=None, temperature=None, max_tokens=None,
                           provider=None, no_thinking=False, reasoning_effort=None):
    """Async version of query_llm. Same interface, returns (clean_text, thinking_text)."""
    settings = load_settings()
    llm_config = settings["llm"]

    model = model or llm_config["default_model"]
    temperature = temperature if temperature is not None else llm_config["temperature"]
    max_tokens = max_tokens or llm_config["max_tokens"]
    retry_attempts = llm_config["retry_attempts"]
    retry_delay = llm_config["retry_delay_seconds"]

    overrides = llm_config.get("model_overrides", {}).get(model, {})
    if overrides.get("max_tokens"):
        max_tokens = max(max_tokens, overrides["max_tokens"])
    if overrides.get("no_thinking"):
        no_thinking = True
    if overrides.get("reasoning_effort"):
        reasoning_effort = overrides["reasoning_effort"]
    reasoning_max_tokens = overrides.get("reasoning_max_tokens")

    if no_thinking and not reasoning_effort:
        reasoning_effort = "none"

    provider = provider or infer_provider(model)

    if "Thinking" in model or "thinking" in model:
        if reasoning_effort != "none":
            max_tokens = max(max_tokens, 16384)

    # Build extra_body for thinking control
    extra_body = None
    if reasoning_max_tokens is not None:
        # Explicit token budget overrides effort (for models like Gemini Pro where effort=none is rejected)
        extra_body = {"reasoning": {"max_tokens": reasoning_max_tokens}}
    elif reasoning_effort:
        if provider == "hyperbolic":
            extra_body = {"enable_thinking": False} if reasoning_effort == "none" else None
        elif provider == "openrouter":
            extra_body = {"reasoning": {"effort": reasoning_effort}}

    if provider == "openai":
        client = get_async_openai_client()
        raw = await async_query_openai_compat(client, prompt, model, temperature, max_tokens, retry_attempts, retry_delay)
    elif provider == "anthropic":
        client = get_async_anthropic_client()
        raw = await async_query_anthropic(client, prompt, model, temperature, max_tokens, retry_attempts, retry_delay)
    elif provider == "hyperbolic":
        client = get_async_hyperbolic_client()
        raw = await async_query_openai_compat(client, prompt, model, temperature, max_tokens, retry_attempts, retry_delay, extra_body=extra_body)
    elif provider == "openrouter":
        client = get_async_openrouter_client()
        raw = await async_query_openai_compat(client, prompt, model, temperature, max_tokens, retry_attempts, retry_delay, extra_body=extra_body)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    clean, thinking = strip_thinking(raw)
    return clean, thinking
