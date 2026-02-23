"""Basic usage example — store and query memories.

Before running, install the package::

    pip install -e .

And configure your ``.env`` file with Neo4j and OpenAI credentials.
"""

from threelane_memory import store, query, close


def main() -> None:
    print("=== Storing memories ===\n")

    ep1 = store("My dog Max is 3 years old and he's a golden retriever", speaker="ankesh")
    print(f"Stored: {ep1}")

    ep2 = store("I work at Google as a software engineer", speaker="ankesh")
    print(f"Stored: {ep2}")

    ep3 = store("My sister Priya lives in Mumbai and she's a doctor", speaker="ankesh")
    print(f"Stored: {ep3}")

    print("\n=== Querying memories ===\n")

    questions = [
        "How old is Max?",
        "Where does my sister live?",
        "What do I do for work?",
    ]

    for q in questions:
        print(f"Q: {q}")
        answer = query(q, speaker="ankesh")
        print(f"A: {answer}\n")

    close()


if __name__ == "__main__":
    main()
