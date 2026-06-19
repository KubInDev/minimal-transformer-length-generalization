"""
data/generators.py
Synthetic formal language generators.

Each function has the signature:
    generate_*(seq_len: int, n_samples: int) -> (X: LongTensor, y: LongTensor)
where X is [n_samples, seq_len] and y is [n_samples].

Task registry at the bottom maps string names to generator functions
used by the experiment runner so you never hard-code task names elsewhere.
"""

import random

import torch


# Task 1: First-Last equality
# Label = 1 if seq[0] == seq[-1]
# Requires non-local endpoint comparison.

def generate_first_last(seq_len: int, n_samples: int):
    X = torch.randint(0, 2, (n_samples, seq_len))
    y = (X[:, 0] == X[:, -1]).long()
    return X, y


# Task 2: A*B* membership
# Valid = all 0s appear before all 1s (any split, including all-0 or all-1).
# Invalid = guaranteed structural violation: at least one 1 before a 0.

def generate_a_star_b_star(seq_len: int, n_samples: int):
    X, y = [], []
    for _ in range(n_samples):
        if random.random() < 0.5:
            # Valid: choose split, 0s on left, 1s on right
            split = random.randint(0, seq_len)
            seq = torch.zeros(seq_len, dtype=torch.long)
            seq[split:] = 1
            label = 1
        else:
            # Invalid: guaranteed 1->0 transition
            # Start from a valid sequence with at least one 0 and one 1
            # then move one 1 from the 1-region into the 0-region.
            split = random.randint(1, seq_len - 1)
            seq = torch.zeros(seq_len, dtype=torch.long)
            seq[split:] = 1
            b_pos = random.randint(0, split - 1)  # destination in 0-region
            a_pos = random.randint(split, seq_len - 1) # source in 1-region
            seq[b_pos] = 1  # 1 inserted before split -> violation
            seq[a_pos] = 0  # 0 preserved after split -> still a 0 after the 1
            label = 0
        X.append(seq)
        y.append(label)
    return torch.stack(X), torch.tensor(y)


# Task 3: Parity
# Label = sum(seq) % 2

def generate_parity(seq_len: int, n_samples: int):
    X = torch.randint(0, 2, (n_samples, seq_len))
    y = X.sum(dim=1) % 2
    return X, y.long()


# Task 4: Contains-11
# Label = 1 if any adjacent pair of 1s exists in the sequence.

def generate_contains_11(seq_len: int, n_samples: int):
    X, y = [], []
    for _ in range(n_samples):
        seq = torch.randint(0, 2, (seq_len,))
        has_11 = ((seq[:-1] == 1) & (seq[1:] == 1)).any().item()
        X.append(seq)
        y.append(int(has_11))
    return torch.stack(X), torch.tensor(y)


# Task 5: Even-Pairs
# Label = 1 if count of 1s is even.


def generate_even_pairs(seq_len: int, n_samples: int):
    X = torch.randint(0, 2, (n_samples, seq_len))
    y = (X.sum(dim=1) % 2 == 0).long()
    return X, y


# Registry
# Add new tasks here - the experiment runner picks them up automatically.

TASK_REGISTRY: dict[str, callable] = {
    "First-Last": generate_first_last,
    "A*B*": generate_a_star_b_star,
    "Parity": generate_parity,
    "Contains-11": generate_contains_11,
    "Even-Pairs": generate_even_pairs,
}


# Sanity check
if __name__ == "__main__":
    print("Generator sanity checks\n" + "-" * 40)
    for name, fn in TASK_REGISTRY.items():
        X, y = fn(20, 1000)
        assert X.shape == (1000, 20), f" Shape error in {name}"
        assert set(y.unique().tolist()).issubset({0, 1}), f" Label error in {name}"
        balance = y.float().mean().item()
        print(f" {name:<15}  shape={tuple(X.shape)}  "
              f"label_mean={balance:.3f}  OK")
    print("\nAll generators OK.")
