"""
eval.py

Run with:
    python eval.py
"""

from rag_pipeline import retrieve_documents, SHARED_SESSION


test_cases = [
    {
        "question": "Who is the manufacturer of the Dassault Rafale?",
        "expected_source": "Dassault_Rafale.pdf",
        "expected_page": 1,
    },
    {
        "question": "When did the Dassault Rafale first enter service?",
        "expected_source": "Dassault_Rafale.pdf",
        "expected_page": 1,
    },
    {
        "question": "Which countries originally collaborated on the Future European Fighter Aircraft project?",
        "expected_source": "Dassault_Rafale.pdf",
        "expected_page": 1,
    },
    {
        "question": "What are the three main variants of the Rafale aircraft?",
        "expected_source": "Dassault_Rafale.pdf",
        "expected_page": 1,
    },
    {
        "question": "What engines power the Dassault Rafale?",
        "expected_source": "Dassault_Rafale.pdf",
        "expected_page": 6,
    },
    {
        "question": "What is the maximum external payload capacity of the Rafale?",
        "expected_source": "Dassault_Rafale.pdf",
        "expected_page": 6,
    },
    {
        "question": "What avionics architecture is used in the Rafale?",
        "expected_source": "Dassault_Rafale.pdf",
        "expected_page": 5,
    },
    {
        "question": "What defensive electronic warfare system protects the Rafale?",
        "expected_source": "Dassault_Rafale.pdf",
        "expected_page": 5,
    },
    {
        "question": "Who manufactures the Dassault Mirage 2000?",
        "expected_source": "Dassault_Mirage_2000.pdf",
        "expected_page": 1,
    },
    {
        "question": "When did the Dassault Mirage 2000 first fly?",
        "expected_source": "Dassault_Mirage_2000.pdf",
        "expected_page": 1,
    },
    {
        "question": "How many Mirage 2000 aircraft were built?",
        "expected_source": "Dassault_Mirage_2000.pdf",
        "expected_page": 1,
    },
    {
        "question": "What engine powers the Mirage 2000?",
        "expected_source": "Dassault_Mirage_2000.pdf",
        "expected_page": 6,
    },
    {
        "question": "What is the primary role of the Mirage 2000N variant?",
        "expected_source": "Dassault_Mirage_2000.pdf",
        "expected_page": 3,
    },
    {
        "question": "What is the wing loading of the Mirage 2000 at a takeoff weight of 15,000 kg?",
        "expected_source": "Dassault_Mirage_2000.pdf",
        "expected_page": 5,
    },
    {
        "question": "What bandwidth is given in the Nyquist and Shannon theorem problem?",
        "expected_source": "AssignmentCN.pdf",
        "expected_page": 1,
    },
    {
        "question": "What is the propagation speed used in the multi-segment end-to-end delay problem?",
        "expected_source": "AssignmentCN.pdf",
        "expected_page": 1,
    },
    {
        "question": "What is the generator polynomial used in the CRC division problem?",
        "expected_source": "AssignmentCN.pdf",
        "expected_page": 2,
    },
    {
        "question": "Which collision resolution technique is specified in the hashing question?",
        "expected_source": "AssignmentADS.pdf",
        "expected_page": 1,
    },
    {
        "question": "Which sequence of numbers is used to construct the Red-Black Tree?",
        "expected_source": "AssignmentADS.pdf",
        "expected_page": 1,
    },
    {
        "question": "Which keys must be deleted from the perfect skip list?",
        "expected_source": "AssignmentADS.pdf",
        "expected_page": 1,
    },
]


def run_eval(top_k: int = 3) -> None:
    hits = 0

    for case in test_cases:
        results = retrieve_documents(case["question"], SHARED_SESSION, top_k=top_k)
        metadatas = results["metadatas"][0]

        found = any(
            m.get("source") == case["expected_source"]
            and m.get("page") == case["expected_page"]
            for m in metadatas
        )

        status = "HIT " if found else "MISS"
        print(f"[{status}] {case['question']}")

        if found:
            hits += 1

    hit_rate = round(hits / len(test_cases) * 100, 1)
    print(f"\nRetrieval hit-rate (top-{top_k}): {hits}/{len(test_cases)} = {hit_rate}%")


if __name__ == "__main__":
    run_eval()
