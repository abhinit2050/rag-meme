"""
Step 7 of the Meme History RAG pipeline: a systematic pass over the decline
path, beyond the ad hoc spot checks used while building steps 5-6.

Three categories are exercised against generate.answer():

- OUT_OF_CORPUS: real memes not in data/meme_list.txt -- should decline.
- UNRELATED: questions with no meme content at all -- should decline.
- IN_CORPUS: control group of questions about seeded memes -- should NOT
  decline, so a passing run means the pipeline is actually discriminating
  rather than declining everything.

Exits non-zero if any case fails, so this can double as a CI-style check.
"""

import sys

from generate import DECLINE_MESSAGE, answer

OUT_OF_CORPUS = [
    "Tell me about the Grumpy Cat meme",
    "What is the origin of Nyan Cat?",
    "Explain the Harlem Shake meme",
    "What is the Ice Bucket Challenge meme about?",
    "Tell me about the This Is Fine dog meme",
    "What's the history of the Big Chungus meme?",
]

UNRELATED = [
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
    "Explain quantum entanglement.",
    "Who won the 2018 World Cup?",
]

IN_CORPUS = [
    "Why is Rickroll called Rickroll?",
    "Who is the dog in the Doge meme?",
    "What is the origin of Pepe the Frog?",
    "Tell me about the Trollface meme.",
]


def run() -> bool:
    all_passed = True

    print("== Out-of-corpus memes (expect decline) ==")
    for q in OUT_OF_CORPUS:
        response = answer(q)
        ok = response == DECLINE_MESSAGE
        all_passed &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {q}")
        if not ok:
            print(f"         got: {response[:200]}")

    print("\n== Unrelated questions (expect decline) ==")
    for q in UNRELATED:
        response = answer(q)
        ok = response == DECLINE_MESSAGE
        all_passed &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {q}")
        if not ok:
            print(f"         got: {response[:200]}")

    print("\n== In-corpus control (expect NOT decline) ==")
    for q in IN_CORPUS:
        response = answer(q)
        ok = response != DECLINE_MESSAGE
        all_passed &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {q}")
        if not ok:
            print(f"         got: {response[:200]}")

    return all_passed


def main() -> None:
    passed = run()
    print(f"\n{'ALL PASSED' if passed else 'SOME FAILED'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
