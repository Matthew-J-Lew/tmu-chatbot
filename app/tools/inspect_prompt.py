import sys
import asyncio

from app.api.db import init_db_pool, close_db_pool, get_pool
from app.rag.retrieval import retrieve
from app.api.main import build_prompt_and_sources


async def _run(question: str, k: int, num_candidates: int) -> None:
    await init_db_pool()
    pool = get_pool()

    chunks = await retrieve(pool=pool, query=question, k=k, num_candidates=num_candidates)
    prompt, sources = build_prompt_and_sources(question, chunks)

    print("\n=== FULL PROMPT SENT TO LLM ===\n")
    print(prompt)

    print("\n=== SOURCES ===")
    for s in sources:
        print(f"[{s.id}] {s.url}")

    await close_db_pool()


def main() -> None:
    # Usage: python -m app.tools.inspect_prompt "question" [k] [num_candidates]
    question = sys.argv[1] if len(sys.argv) > 1 else "What undergraduate programs are offered at the Faculty of Arts?"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    num_candidates = int(sys.argv[3]) if len(sys.argv) > 3 else 12

    asyncio.run(_run(question, k, num_candidates))


if __name__ == "__main__":
    main()
