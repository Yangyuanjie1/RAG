from openai import AuthenticationError, OpenAI

from app.core.config import settings


def answer_with_context(question: str, sources: list[dict[str, str | int]]) -> str:
    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env.")

    if not sources:
        return "I could not find relevant content in the uploaded documents. Try uploading more relevant material or ask a more specific question."

    context_blocks = []
    for index, source in enumerate(sources, start=1):
        context_blocks.append(
            f"[Source {index}] file={source['document_name']} chunk={source['chunk_id']}\n{source['content']}"
        )

    system_prompt = (
        "You are a RAG assistant for document question answering. "
        "Answer only from the provided context. "
        "If the context is insufficient, say so explicitly. "
        "At the end, briefly reference the source file names you used."
    )
    user_prompt = f"Question:\n{question}\n\nContext:\n" + "\n\n".join(context_blocks)

    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except AuthenticationError as exc:
        raise RuntimeError("MiniMax authentication failed. Check OPENAI_API_KEY.") from exc
    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    if not response.choices or not response.choices[0].message.content:
        raise RuntimeError("Model returned an empty response.")

    return response.choices[0].message.content
