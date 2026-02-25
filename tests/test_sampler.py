from model.dataset import PKSampler


class TestPKSampler:
    def test_batch_size_is_p_times_k(self):
        labels = [c for c in range(20) for _ in range(5)]
        sampler = PKSampler(labels, p=4, k=3, num_batches=10)
        for batch in sampler:
            assert len(batch) == 12  # 4 * 3

    def test_each_batch_has_p_distinct_classes(self):
        labels = [c for c in range(20) for _ in range(5)]
        sampler = PKSampler(labels, p=4, k=3, num_batches=10)
        for batch in sampler:
            classes_in_batch = {labels[i] for i in batch}
            assert len(classes_in_batch) == 4

    def test_len_matches_num_batches(self):
        labels = list(range(100))
        sampler = PKSampler(labels, p=4, k=2, num_batches=7)
        assert len(sampler) == 7
        assert sum(1 for _ in sampler) == 7

    def test_fewer_classes_than_p_does_not_crash(self):
        # Only 3 classes but p=8 — should clamp to available classes
        labels = [c for c in range(3) for _ in range(4)]
        sampler = PKSampler(labels, p=8, k=2, num_batches=5)
        for batch in sampler:
            assert len(batch) > 0

    def test_all_yielded_indices_are_valid(self):
        labels = [c for c in range(10) for _ in range(3)]
        sampler = PKSampler(labels, p=4, k=2, num_batches=5)
        n = len(labels)
        for batch in sampler:
            assert all(0 <= i < n for i in batch)

    def test_default_num_batches_matches_dataset_size(self):
        # Default: len(labels) // (p * k)
        labels = [c for c in range(20) for _ in range(5)]  # 100 samples
        sampler = PKSampler(labels, p=4, k=2)
        assert len(sampler) == 100 // 8  # 12
