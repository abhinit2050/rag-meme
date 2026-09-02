"""
Retrieval eval for the Meme History RAG pipeline.

Checks whether retrieve() (step 4) surfaces chunks from the correct meme for
a golden set of meme-specific questions -- this is distinct from
test_decline.py, which exercises step 5's grounding/decline behavior rather
than retrieval quality itself.

Reports, for k results per query:
- Recall@k:    did the correct meme show up anywhere in the top k?
- Precision@k: how much of the top k actually belongs to the correct meme?
- MRR:         how high did the first correct hit rank, on average?
"""

import sys

from retrieve import retrieve

K = 5

# (question, expected meme_slug) -- two phrasings per seed meme so a single
# lucky/unlucky match doesn't dominate the score.
GOLDEN = [
    ("Why is Rickroll called Rickroll?", "rickroll"),
    ("What song is used in a rickroll?", "rickroll"),
    ("What is the Distracted Boyfriend meme about?", "distracted-boyfriend"),
    ("Who took the original Distracted Boyfriend photo?", "distracted-boyfriend"),
    ("Who is the dog in the Doge meme?", "doge"),
    ("What font is typically used in Doge captions?", "doge"),
    ("What is the origin of Pepe the Frog?", "pepe-the-frog"),
    ("Who created Pepe the Frog?", "pepe-the-frog"),
    ("What is the Trollface meme?", "trollface"),
    ("Who drew the original Trollface?", "trollface"),
    ("What is the Bad Luck Brian meme about?", "bad-luck-brian"),
    ("Who is the person in the Bad Luck Brian photo?", "bad-luck-brian"),
    ("What is the Success Kid meme?", "success-kid-i-hate-sandcastles"),
    ("Where did the Success Kid photo come from?", "success-kid-i-hate-sandcastles"),
    ("What is the Overly Attached Girlfriend meme?", "overly-attached-girlfriend"),
    ("Who is the woman in the Overly Attached Girlfriend photo?", "overly-attached-girlfriend"),
]


def evaluate(k: int = K) -> bool:
    n = len(GOLDEN)
    hits = 0
    precisions = []
    reciprocal_ranks = []

    print(f"Retrieval eval -- {n} questions, k={k}\n")
    for question, expected_slug in GOLDEN:
        results = retrieve(question, k=k)
        slugs = [r["meme_slug"] for r in results]

        hit = expected_slug in slugs
        hits += hit

        precision = slugs.count(expected_slug) / k
        precisions.append(precision)

        rank = next((i + 1 for i, s in enumerate(slugs) if s == expected_slug), None)
        reciprocal_ranks.append(1 / rank if rank else 0)

        print(f"  [{'PASS' if hit else 'FAIL'}] {question}")
        if not hit:
            print(f"         expected={expected_slug}  got={slugs}")

    recall_at_k = hits / n
    mean_precision = sum(precisions) / n
    mrr = sum(reciprocal_ranks) / n

    print(f"\nRecall@{k}:    {recall_at_k:.2%}  ({hits}/{n})")
    print(f"Precision@{k}: {mean_precision:.2%}")
    print(f"MRR:          {mrr:.3f}")

    return hits == n


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else K
    passed = evaluate(k)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
