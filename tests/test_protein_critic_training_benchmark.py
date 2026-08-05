from scripts.benchmark_protein_critic_training import (
    candidate_batches,
    stratified_indices,
)


class LengthDataset:
    def __init__(self, lengths):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def sequence_length(self, index):
        return self.lengths[index]


def test_stratified_indices_include_length_endpoints():
    dataset = LengthDataset([50, 10, 40, 20, 30])
    selected = stratified_indices(dataset, 3)
    assert [dataset.sequence_length(index) for index in selected] == [10, 30, 50]


def test_candidate_batches_preserve_same_sample():
    indices = list(range(8))
    batches = candidate_batches(indices, 3)
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7]]
    assert [index for batch in batches for index in batch] == indices
