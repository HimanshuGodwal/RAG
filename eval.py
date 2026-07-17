"""
eval.py

Run with:
    python eval.py
"""

from rag_pipeline import retrieve_documents, SHARED_SESSION


test_cases = [
    {
        "question": "What is the primary mission of the Dassault Rafale?",
        "expected_source": "Dassault_Rafale.pdf",
        "expected_page": 1,
    },
    {
        "question": "Which air forces currently operate the Dassault Rafale?",
        "expected_source": "Dassault_Rafale.pdf",
        "expected_page": 1,
    },
    {
        "question": "What countries originally collaborated on the Future European Fighter Aircraft project?",
        "expected_source": "Dassault_Rafale.pdf",
        "expected_page": 1,
    },
    {
        "question": "What are the three main variants of the Rafale aircraft?",
        "expected_source": "Dassault_Rafale.pdf",
        "expected_page": 1,
    },
    {
        "question": "Which engines power the Dassault Rafale?",
        "expected_source": "Dassault_Rafale.pdf",
        "expected_page": 6,
    },
    {
        "question": "What is the maximum external payload capacity of the Rafale?",
        "expected_source": "Dassault_Rafale.pdf",
        "expected_page": 6,
    },
    {
        "question": "Which avionics architecture is integrated into the Rafale?",
        "expected_source": "Dassault_Rafale.pdf",
        "expected_page": 5,
    },
    {
        "question": "What electronic warfare system provides self-protection for the Rafale?",
        "expected_source": "Dassault_Rafale.pdf",
        "expected_page": 5,
    },
    {
        "question": "What type of aircraft is the Dassault Mirage 2000?",
        "expected_source": "Dassault_Mirage_2000.pdf",
        "expected_page": 1,
    },
    {
        "question": "Which air forces are listed as the primary operators of the Mirage 2000?",
        "expected_source": "Dassault_Mirage_2000.pdf",
        "expected_page": 1,
    },
    {
        "question": "How many Mirage 2000 aircraft were built?",
        "expected_source": "Dassault_Mirage_2000.pdf",
        "expected_page": 1,
    },
    {
        "question": "Which engine powers the Mirage 2000 fighter aircraft?",
        "expected_source": "Dassault_Mirage_2000.pdf",
        "expected_page": 6,
    },
    {
        "question": "What is the purpose of the Mirage 2000N variant?",
        "expected_source": "Dassault_Mirage_2000.pdf",
        "expected_page": 3,
    },
    {
        "question": "What is the wing loading of the Mirage 2000 at a takeoff weight of 15,000 kg?",
        "expected_source": "Dassault_Mirage_2000.pdf",
        "expected_page": 5,
    },
    {
        "question": "According to the networking assignment, what is the bandwidth of the communication channel in Question 1?",
        "expected_source": "AssignmentCN.pdf",
        "expected_page": 1,
    },
    {
        "question": "What propagation speed is specified for the multi-hop transmission problem?",
        "expected_source": "AssignmentCN.pdf",
        "expected_page": 1,
    },
    {
        "question": "Which generator polynomial is used in the CRC division problem?",
        "expected_source": "AssignmentCN.pdf",
        "expected_page": 2,
    },
    {
        "question": "Which collision resolution method is required in the hashing problem?",
        "expected_source": "AssignmentADS.pdf",
        "expected_page": 1,
    },
    {
        "question": "What sequence of values is used to construct the Red-Black Tree?",
        "expected_source": "AssignmentADS.pdf",
        "expected_page": 1,
    },
    {
        "question": "Which elements are removed from the perfect skip list after construction?",
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
