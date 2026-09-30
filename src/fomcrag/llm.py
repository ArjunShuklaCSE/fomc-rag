"""One chat() function over two backends.

    local      Qwen3-4B-Instruct-2507, 4-bit NF4 on the GPU (default: free, reproducible, ~3 GB VRAM)
    anthropic  Claude via the Anthropic API when ANTHROPIC_API_KEY is set (or LLM_PROVIDER=anthropic)

Reported answer-quality numbers name the backend they were produced with.
"""

import os
import re
from functools import lru_cache

from .models import GENERATOR, resolve

CLAUDE_MODEL = os.environ.get("FOMC_CLAUDE_MODEL", "claude-opus-5-5")


def provider() -> str:
    return os.environ.get("LLM_PROVIDER") or ("anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "local")


def backend_name() -> str:
    return CLAUDE_MODEL if provider() == "anthropic" else GENERATOR + " (4-bit)"


@lru_cache(maxsize=1)
def _local():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    path = resolve(GENERATOR)
    tok = AutoTokenizer.from_pretrained(path)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(path, quantization_config=quant, device_map="cuda:0",
                                                 dtype=torch.float16)
    return tok, model


def _chat_local(system: str, user: str, max_tokens: int) -> str:
    import torch
    tok, model = _local()
    prompt = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                     tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@lru_cache(maxsize=1)
def _client():
    import anthropic
    return anthropic.Anthropic()


def _chat_anthropic(system: str, user: str, max_tokens: int) -> str:
    response = _client().beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max(max_tokens, 4000),  # headroom for thinking, which cannot be disabled on this model
        system=system,
        output_config={"effort": "low"},   # grounded QA over given passages: depth buys little
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",                # a policy decline is retried on a fallback model server-side
        messages=[{"role": "user", "content": user}],
    )
    if response.stop_reason == "refusal":
        return "I don't know."
    return "".join(b.text for b in response.content if b.type == "text").strip()


def chat(system: str, user: str, max_tokens: int = 400) -> str:
    return _chat_anthropic(system, user, max_tokens) if provider() == "anthropic" else _chat_local(system, user, max_tokens)


HYDE_SYSTEM = ("You write short passages in the style of Federal Open Market Committee minutes and press conference "
               "transcripts. Given a question, write the 2-3 sentence passage that would answer it. Invent plausible "
               "specifics if needed; the passage is only used as a search query.")


@lru_cache(maxsize=512)
def hyde(query: str) -> str:
    """Hypothetical Document Embeddings: embed an imagined answer passage instead of the bare question."""
    passage = chat(HYDE_SYSTEM, query, max_tokens=120)
    return re.sub(r"\s+", " ", f"{query} {passage}").strip()
