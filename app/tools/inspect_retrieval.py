import sys
import asyncio

from app.api.db import init_db_pool, close_db_pool, get_pool
from app.rag.retrieval import retrieve


async def _run(question: str, k: int, num_candidates: int, max_chars: int) -> None:
    await init_db_pool()
    pool = get_pool()

    chunks = await retrieve(pool=pool, query=question, k=k, num_candidates=num_candidates)

    print("\n=== RETRIEVED CHUNKS ===")
    print(f"question: {question}")
    print(f"k={k} num_candidates={num_candidates}\n")

    for i, c in enumerate(chunks, 1):
        url = c.get("chunk_url") or c.get("source_url") or ""
        section = c.get("section")
        text = c.get("chunk") or ""
        if max_chars and len(text) > max_chars:
            text = text[: max_chars - 3] + "..."

        print(f"\n--- CHUNK [{i}] ---")
        print(f"URL: {url}")
        print(f"Section: {section}")
        print("Text:\n" + text)

    await close_db_pool()


def main() -> None:
    # Usage: python -m app.tools.inspect_retrieval "question" [k] [num_candidates] [max_chars]
    question = sys.argv[1] if len(sys.argv) > 1 else "What undergraduate programs are offered at the Faculty of Arts?"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    num_candidates = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    max_chars = int(sys.argv[4]) if len(sys.argv) > 4 else 1500  # preview cap

    asyncio.run(_run(question, k, num_candidates, max_chars))


if __name__ == "__main__":
    main()
