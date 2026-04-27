"""
LLM response generation.

Builds a grounded prompt from retrieved context and calls the OpenAI
chat completion API. Includes token budget management to avoid
exceeding the context window.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.models.schemas import SourceChunk
from app.utils.helpers import count_tokens, truncate_to_tokens
from app.utils.logger import LoggerMixin, log_event

settings = get_settings()

# ── Prompt Templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise and helpful AI assistant. Answer the user's question \
based ONLY on the provided context. If the context does not contain enough information \
to answer confidently, say so clearly. Do not fabricate information.

Guidelines:
- Be concise and accurate.
- Reference the context directly when appropriate.
- If multiple sources are relevant, synthesise them coherently.
- If the answer is not in the context, respond: "I don't have enough information \
in the provided documents to answer this question."
"""

CONTEXT_TEMPLATE = """Relevant document excerpts:
{context}

---
Question: {question}

Answer:"""


# ── Response Dataclass ────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


# ── Generator ─────────────────────────────────────────────────────────────────

class ResponseGenerator(LoggerMixin):
    """
    Wraps OpenAI chat completions.

    Token budget
    ------------
    Total context window (128k for gpt-4o-mini) is split as:
      - System prompt   ~  300 tokens
      - Retrieved docs  ~ 6000 tokens  (trimmed if needed)
      - User question   ~  500 tokens
      - Answer          ~ LLM_MAX_TOKENS (default 1024)

    If the combined context exceeds the budget, older chunks are dropped
    until it fits.
    """

    CONTEXT_TOKEN_BUDGET = 6_000
    MAX_CHUNK_TOKENS = 800  # trim individual long chunks to this

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.llm_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature

    # ── Public API ────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def generate(
        self, question: str, chunks: list[SourceChunk]
    ) -> GenerationResult:
        """
        Generate a grounded answer given a question and retrieved chunks.

        Parameters
        ----------
        question : str
            The user's original question.
        chunks : list[SourceChunk]
            Retrieved, ranked context chunks.

        Returns
        -------
        GenerationResult with the answer text and token usage.
        """
        context_str = self._build_context(chunks)
        user_prompt = CONTEXT_TEMPLATE.format(
            context=context_str, question=question
        )

        log_event(
            self.logger,
            "generation_start",
            model=self._model,
            context_tokens=count_tokens(context_str),
            chunks=len(chunks),
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = response.choices[0].message.content or ""
        usage = response.usage

        log_event(
            self.logger,
            "generation_complete",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

        return GenerationResult(
            answer=answer.strip(),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            model=self._model,
        )

    # ── Context Building ──────────────────────────────────────────────

    def _build_context(self, chunks: list[SourceChunk]) -> str:
        """
        Concatenate chunk excerpts up to the token budget.
        Each chunk is prefixed with source info for attribution.
        """
        parts: list[str] = []
        used_tokens = 0

        for i, chunk in enumerate(chunks, start=1):
            header = f"[Source {i}: {chunk.document}"
            if chunk.page:
                header += f", page {chunk.page}"
            header += "]"

            body = truncate_to_tokens(chunk.excerpt, self.MAX_CHUNK_TOKENS)
            block = f"{header}\n{body}"
            block_tokens = count_tokens(block)

            if used_tokens + block_tokens > self.CONTEXT_TOKEN_BUDGET:
                self.logger.debug(
                    f"Token budget reached at chunk {i}/{len(chunks)} — stopping."
                )
                break

            parts.append(block)
            used_tokens += block_tokens

        return "\n\n".join(parts)
