"""Build the main (WMDP-bio cloze) and control (MMLU bio/med) prompt sets.

Design decisions implemented here are fixed in docs/plan.md T2 (probe design,
split conventions) and T3 (item sets and split construction).

Prompt that produced this script:

    Write scripts/build_prompt_sets.py. Two outputs, both written to data/ as
    JSON with the seed recorded inside.

    Main set: load EleutherAI/wmdp_bio_cloze split cloze_compatible (1076
    items). Split into 70% train and 30% held-out at the item level with seed
    42, so an item and all four of its candidate answers stay on the same
    side. Record the item indices for each side.

    Control set: load cais/mmlu for these eight subjects: college_biology,
    high_school_biology, virology, anatomy, clinical_knowledge,
    medical_genetics, college_medicine, professional_medicine. Take a random
    1076 of them with seed 42. Reformat each into the same shape as the cloze
    items: a question string, a choices list of four strings, and an answer
    index. Split 70/30 the same way.

    Print the counts and confirm no overlap between train and held-out for
    either set.

Seed convention (revised after review): SEED = 42 for the main split and the
control sampling; the control train/held-out split uses SEED + 1 = 43 so the
two sets' index partitions are visibly independent. Both seeds are recorded
in control_set.json (sampling_seed, split_seed).

Run as: .venv/bin/python scripts/build_prompt_sets.py
Outputs: data/main_set.json, data/control_set.json
"""

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

SEED = 42
TRAIN_FRAC = 0.7
N_CONTROL = 1076
DATA_DIR = Path("data")

MMLU_SUBJECTS = [
    "college_biology",
    "high_school_biology",
    "virology",
    "anatomy",
    "clinical_knowledge",
    "medical_genetics",
    "college_medicine",
    "professional_medicine",
]


def split_indices(n: int, seed: int) -> tuple[list[int], list[int]]:
    """Item-level 70/30 split: permute [0, n) with the seed, cut at 70%."""
    perm = np.random.default_rng(seed).permutation(n)
    n_train = round(TRAIN_FRAC * n)
    return sorted(perm[:n_train].tolist()), sorted(perm[n_train:].tolist())


def check_and_report(name: str, items: list, train: list[int], heldout: list[int]) -> None:
    overlap = set(train) & set(heldout)
    assert not overlap, f"{name}: train/held-out overlap: {sorted(overlap)[:10]}"
    assert len(train) + len(heldout) == len(items)
    assert all(len(it["choices"]) == 4 for it in items), f"{name}: item without 4 choices"
    assert all(it["answer"] in (0, 1, 2, 3) for it in items), f"{name}: answer index out of range"
    print(
        f"{name}: {len(items)} items -> train={len(train)}, "
        f"heldout={len(heldout)}, overlap={len(overlap)}"
    )


def build_main() -> dict:
    ds = load_dataset("EleutherAI/wmdp_bio_cloze", split="cloze_compatible")
    items = [
        {"question": ex["question"], "choices": ex["choices"], "answer": ex["answer"]}
        for ex in ds
    ]
    train, heldout = split_indices(len(items), SEED)
    check_and_report("main (wmdp_bio_cloze)", items, train, heldout)
    return {
        "seed": SEED,
        "source": "EleutherAI/wmdp_bio_cloze, split cloze_compatible",
        "train_frac": TRAIN_FRAC,
        "train_indices": train,
        "heldout_indices": heldout,
        "items": items,
    }


def build_control() -> dict:
    # Concatenate the eight subjects' test splits in the fixed order above;
    # provenance of each item is (subject, row index within that test split).
    pool = []
    for subject in MMLU_SUBJECTS:
        ds = load_dataset("cais/mmlu", subject, split="test")
        for row, ex in enumerate(ds):
            pool.append(
                {
                    "question": ex["question"],
                    "choices": ex["choices"],
                    "answer": int(ex["answer"]),  # ClassLabel -> plain int
                    "subject": subject,
                    "source_row": row,
                }
            )
    print(f"control pool: {len(pool)} items from {len(MMLU_SUBJECTS)} subjects")

    keep = np.random.default_rng(SEED).choice(len(pool), size=N_CONTROL, replace=False)
    items = [pool[i] for i in sorted(keep.tolist())]
    train, heldout = split_indices(len(items), SEED + 1)
    check_and_report("control (mmlu bio/med)", items, train, heldout)
    return {
        "sampling_seed": SEED,
        "split_seed": SEED + 1,
        "source": f"cais/mmlu test splits, subjects={MMLU_SUBJECTS}",
        "train_frac": TRAIN_FRAC,
        "pool_size": len(pool),
        "train_indices": train,
        "heldout_indices": heldout,
        "items": items,
    }


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for fname, payload in [
        ("main_set.json", build_main()),
        ("control_set.json", build_control()),
    ]:
        out = DATA_DIR / fname
        out.write_text(json.dumps(payload, indent=1))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
