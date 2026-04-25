"""
pipeline.py
───────────
The end-to-end question → answer flow.

Steps:
  1. Classify the query (router's classify_query)
  2. Embed the query
  3. Search vector store
  4. Route/re-rank results
  5. Build prompt with context
  6. Call LLM
  7. Return structured response with source citations
"""

from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.storage.embedder import Embedder
from src.storage.vector_store import VectorStore
from src.router.router_model import RouterModel, ScoredChunk
from src.utils.config_loader import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)

RETRYABLE_EXCEPTIONS = (TimeoutError, ConnectionError)

@dataclass
class QueryResponse:
    """Structured response from the pipeline."""
    question:     str
    answer:       str
    sources:      list[dict]    # [{title, url, timestamp_link, similarity}]
    query_type:   str
    chunk_count:  int           # How many chunks were used
    model_used:   str

# Input Validation

def _validate_question(question: str) -> str:
    if not isinstance(question, str):
        raise TypeError("Question must be a string")

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty")

    if len(question) > cfg.query.max_query_len:
        raise ValueError("Question too long")

    return question


# Prompt Builder (Injection-aware)

def _sanitize_text(text: str) -> str:
    """Basic sanitization to reduce prompt injection risk."""
    return text.replace("{", "").replace("}", "")


def _build_context_prompt(query: str, chunks: list[ScoredChunk], query_type: str) -> str:
    """
    Build the system + user prompt sent to the LLM.

    Context is prepended in order of relevance (best first).
    Each chunk is labeled with its source for attribution.
    """

    query = _sanitize_text(query)

    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        mins = int(chunk.start_time // 60)
        secs = int(chunk.start_time % 60)

        safe_text = _sanitize_text(chunk.text)

        context_blocks.append(
            f"[Source {i}] Video: \"{chunk.video_title}\" | "
            f"Timestamp: {mins}:{secs:02d}\n"
            f"{safe_text}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    # Adjust instructions based on query type
    type_instructions = {
        "factual":     "Give a direct, precise answer. Cite which source(s) you used.",
        "comparative": "Compare and contrast the information across sources.",
        "summary":     "Provide a comprehensive overview using all relevant sources.",
        "procedural":  "List clear steps. Note which video each step comes from.",
        "open":        "Answer thoroughly using the provided context.",
    }
    instruction = type_instructions.get(query_type, type_instructions["open"])

    return f"""You are a helpful assistant answering questions about YouTube video content.
Use ONLY the context provided below to answer. If the answer isn't in the context, say so clearly.
{instruction}

Always end your answer with a "Sources:" section listing the video titles and timestamps you referenced.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

@retry(
    stop=stop_after_attempt(cfg.llm.max_retries),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    reraise=True,
)
def _call_cerebras(prompt: str) -> tuple[str, str]:
    """Call Cerebras's API. Returns (response_text, model_name)."""
    from cerebras.cloud.sdk import Cerebras

    api_key = cfg.secrets.cerebras_api_key
    if not api_key:
        raise ValueError("Cerebras_API_KEY not set in .env file. ")

    client = Cerebras(api_key=api_key)
    model  = cfg.llm.model

    response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=cfg.llm.max_tokens,
            temperature=cfg.llm.temperature,
        )

    content = response.choices[0].message.content

    if not content:
        raise ValueError(f"Empty LLM ({model}) response")
    
    return content.strip(), model


class QueryPipeline:
    """
    Orchestrates the full question → answer pipeline.

    Initialize once and reuse for multiple queries (embedder and
    vector store have expensive startup costs).
    """

    def __init__(self):
        logger.info("Initializing query pipeline...")
        self.embedder     = Embedder()
        self.vector_store = VectorStore()
        self.router       = RouterModel()
        logger.info("Pipeline ready")

    def ask(self, question: str) -> QueryResponse:
        """
        Process a user question and return an answer with sources.

        Args:
            question: Natural language question string

        Returns:
            QueryResponse with answer and cited sources
        """

        # Validate input
        try:
            question = _validate_question(question)
            
            logger.info("Processing question ...")

            # ── Step 1: Classify the query ─────────────────────────────
            query_type = self.router.classify_query(question)
            logger.debug(f"Query type: {query_type}")

            # ── Step 2: Embed the query ────────────────────────────────
            query_embedding = self.embedder.embed_query(question)

            # ── Step 3: Retrieve candidates from vector store ──────────
            n_candidates = cfg.vector_store.n_results
            candidates   = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                n_results=n_candidates,
            )

            if not candidates:
                return QueryResponse(
                    question=question,
                    answer="I couldn't find any relevant content. Try re-indexing the channel.",
                    sources=[],
                    query_type=query_type,
                    chunk_count=0,
                    model_used="none",
                )

            # ── Step 4: Route and re-rank ──────────────────────────────
            top_chunks = self.router.route(question, candidates)

            if not top_chunks:
                return QueryResponse(
                    question=question,
                    answer=(
                        "I found some content but none seemed relevant enough to your question. "
                        "Try rephrasing or asking something more specific."
                    ),
                    sources=[],
                    query_type=query_type,
                    chunk_count=0,
                    model_used="none",
                )

            # ── Step 5: Build prompt ───────────────────────────────────
            prompt = _build_context_prompt(question, top_chunks, query_type)
            logger.debug(f"Prompt built with {len(top_chunks)} chunks")

            # ── Step 6: Call LLM ───────────────────────────────────────
            provider = cfg.llm.provider
            try:
                if provider == "cerebras":
                    answer, model_name = _call_cerebras(prompt)
                else:
                    raise ValueError(f"Unknown LLM provider: {provider}. Use 'cerebras'")
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                raise

            # ── Step 7: Build response with citations ──────────────────
            sources = [
                {
                    "title":          chunk.video_title,
                    "url":            chunk.timestamp_link,
                    "similarity":     round(chunk.similarity, 3),
                    "start_time":     chunk.start_time,
                }
                for chunk in top_chunks
            ]

            return QueryResponse(
                question=question,
                answer=answer,
                sources=sources,
                query_type=query_type,
                chunk_count=len(top_chunks),
                model_used=model_name,
            )
        
        except Exception as e:
            logger.exception("Query pipeline failed")
            raise