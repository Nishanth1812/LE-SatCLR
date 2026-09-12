import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.dashboard import select_evaluation_indices


def test_select_evaluation_indices_draws_a_unique_subset():
    indices = [10, 20, 30, 40]

    selected = select_evaluation_indices(indices, 3)

    assert len(selected) == 3
    assert len(set(selected)) == 3
    assert set(selected).issubset(indices)


def test_select_evaluation_indices_keeps_full_split_for_zero_limit():
    indices = [10, 20, 30]

    assert select_evaluation_indices(indices, 0) == indices
