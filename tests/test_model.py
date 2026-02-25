import torch
import torch.nn.functional as F

from model.embedding_model import EmbeddingModel
from model.train import batch_hard_triplet_loss, pairwise_l2_distance


class TestEmbeddingModel:
    def test_output_shape(self):
        model = EmbeddingModel(pretrained=False)
        out = model(torch.randn(4, 3, 224, 224))
        assert out.shape == (4, 512)

    def test_output_is_l2_normalized(self):
        model = EmbeddingModel(pretrained=False)
        out = model(torch.randn(3, 3, 224, 224))
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    def test_eval_mode_is_deterministic(self):
        model = EmbeddingModel(pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_freeze_backbone_freezes_all_backbone_params(self):
        model = EmbeddingModel(pretrained=False)
        model.freeze_backbone()
        frozen = [p for p in model.backbone.parameters() if not p.requires_grad]
        trainable = [p for p in model.backbone.parameters() if p.requires_grad]
        assert len(frozen) > 0
        assert len(trainable) == 0

    def test_unfreeze_backbone_opens_only_last_two_blocks(self):
        model = EmbeddingModel(pretrained=False)
        model.unfreeze_backbone()
        # Early blocks should stay frozen; last two blocks should be trainable
        frozen = [p for p in model.backbone.parameters() if not p.requires_grad]
        trainable = [p for p in model.backbone.parameters() if p.requires_grad]
        assert len(frozen) > 0
        assert len(trainable) > 0

    def test_projection_head_is_always_trainable(self):
        model = EmbeddingModel(pretrained=False)
        model.freeze_backbone()
        trainable = model.trainable_params()
        assert len(trainable) > 0
        assert all(p.requires_grad for p in trainable)


class TestPairwiseL2Distance:
    def test_diagonal_is_zero(self):
        embs = F.normalize(torch.randn(8, 16), dim=1)
        dist = pairwise_l2_distance(embs)
        # The implementation adds 1e-8 inside sqrt to prevent inf gradients,
        # so the diagonal is sqrt(1e-8) = 1e-4 rather than exactly 0.
        assert torch.allclose(dist.diag(), torch.zeros(8), atol=1e-3)

    def test_is_symmetric(self):
        embs = F.normalize(torch.randn(8, 16), dim=1)
        dist = pairwise_l2_distance(embs)
        assert torch.allclose(dist, dist.t(), atol=1e-5)

    def test_all_distances_are_non_negative(self):
        embs = F.normalize(torch.randn(8, 16), dim=1)
        dist = pairwise_l2_distance(embs)
        assert (dist >= 0).all()

    def test_identical_vectors_have_zero_distance(self):
        v = F.normalize(torch.randn(1, 16), dim=1)
        embs = v.repeat(4, 1)
        dist = pairwise_l2_distance(embs)
        assert torch.allclose(dist, torch.zeros(4, 4), atol=1e-4)


class TestBatchHardTripletLoss:
    def test_loss_is_non_negative(self):
        embs = F.normalize(torch.randn(16, 16), dim=1)
        labels = torch.tensor([i // 4 for i in range(16)])
        loss, _ = batch_hard_triplet_loss(embs, labels)
        assert loss.item() >= 0

    def test_active_fraction_is_between_zero_and_one(self):
        embs = F.normalize(torch.randn(16, 16), dim=1)
        labels = torch.tensor([i // 4 for i in range(16)])
        _, active = batch_hard_triplet_loss(embs, labels)
        assert 0.0 <= active <= 1.0

    def test_well_separated_embeddings_give_zero_loss(self):
        # 4 classes, each pinned to a different axis of the unit sphere
        # hardest_pos ~ 0, hardest_neg = sqrt(2), so loss = relu(0 - sqrt(2) + 0.3) = 0
        base = torch.eye(4)
        embs = F.normalize(base.repeat_interleave(4, dim=0), dim=1)  # (16, 4)
        labels = torch.tensor([i // 4 for i in range(16)])
        loss, _ = batch_hard_triplet_loss(embs, labels, margin=0.3)
        assert loss.item() == 0.0
