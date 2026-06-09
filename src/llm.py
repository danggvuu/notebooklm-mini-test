"""
LLM Factory Pattern — Tầng Tạo sinh
Supports 3 backends: vLLM (Production), HuggingFace (Dev), Gemini (Baseline).
Provides both synchronous invoke and async streaming interfaces.
"""
from functools import lru_cache
from typing import List, Optional, Any, Iterator
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_openai import ChatOpenAI
from src.config import settings


class LegacyGeminiChat(BaseChatModel):
    model_name: str
    temperature: float
    api_key: str

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> ChatResult:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        prompt = messages[-1].content
        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature
        )
        res = model.generate_content(prompt, generation_config=generation_config)

        ai_msg = AIMessage(content=res.text)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    def stream_text(self, prompt: str) -> Iterator[str]:
        """Stream tokens from Gemini."""
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature
        )
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            stream=True,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    @property
    def _llm_type(self) -> str:
        return "legacy-gemini-chat"


# ---------------------------------------------------------------------------
# LLM Factory — Build functions
# ---------------------------------------------------------------------------

def _build_hf_local():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    tokenizer = AutoTokenizer.from_pretrained(settings.hf_model)
    model = AutoModelForCausalLM.from_pretrained(
        settings.hf_model,
        dtype=torch.bfloat16,
        device_map=settings.hf_device
    )

    text_gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=settings.hf_device,
        return_full_text=False,
    )
    text_gen.generation_config.max_new_tokens = settings.hf_max_new_tokens
    text_gen.generation_config.do_sample = settings.llm_temperature > 0

    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=text_gen))


def _build_gemini():
    return LegacyGeminiChat(
        model_name=settings.gemini_model,
        temperature=settings.llm_temperature,
        api_key=settings.google_api_key,
    )


def _build_vllm():
    return ChatOpenAI(
        model=settings.hf_model,
        openai_api_key=settings.vllm_api_key,
        openai_api_base=settings.vllm_api_base,
        temperature=settings.llm_temperature,
    )


# ---------------------------------------------------------------------------
# LLM Factory — Public API
# ---------------------------------------------------------------------------

_BUILDERS = {
    "hf_local": _build_hf_local,
    "gemini": _build_gemini,
    "vllm": _build_vllm,
}


@lru_cache(maxsize=4)
def get_llm(provider=None) -> BaseChatModel:
    """
    LLM Factory Pattern.
    Returns the appropriate LLM instance based on provider config.
    """
    provider = provider or settings.llm_provider
    builder = _BUILDERS.get(provider)
    if builder is None:
        raise ValueError(f"Unknown llm_provider '{provider}'. Available: {list(_BUILDERS.keys())}")
    return builder()


def invoke_llm(prompt: str, provider=None) -> str:
    """Invoke LLM synchronously. Returns the full response text."""
    response = get_llm(provider=provider).invoke([HumanMessage(content=prompt)])
    return response.content if isinstance(response.content, str) else str(response.content)


def stream_llm(prompt: str, provider=None) -> Iterator[str]:
    """
    Stream LLM response token by token.
    Falls back to single-shot invoke if streaming is not supported.
    """
    llm = get_llm(provider=provider)

    # Use native streaming if available (Gemini)
    if isinstance(llm, LegacyGeminiChat):
        yield from llm.stream_text(prompt)
        return

    # For vLLM (ChatOpenAI) — use LangChain streaming
    if hasattr(llm, 'stream'):
        try:
            for chunk in llm.stream([HumanMessage(content=prompt)]):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if content:
                    yield content
            return
        except Exception:
            pass  # Fall through to non-streaming

    # Fallback: invoke and yield all at once
    yield invoke_llm(prompt, provider=provider)
