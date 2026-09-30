import torch

from acsi_bqal import (
    AdaptiveCrossScaleInteraction,
    BoundaryQualityHead,
    boundary_quality_targets,
    calibrate_scores,
)


def test_acsi_preserves_shape_and_normalizes_scale_weights():
    module = AdaptiveCrossScaleInteraction(channels=32, interaction_dim=16)
    x = torch.randn(2, 32, 48)
    output, weights, attention = module(x, return_weights=True)
    assert output.shape == x.shape
    assert weights.shape == (2, 48, 3)
    assert attention.shape == (2, 48, 3, 3)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 48))
    torch.testing.assert_close(attention.sum(-1), torch.ones(2, 48, 3))


def test_bqal_sampling_and_calibration_are_finite():
    head = BoundaryQualityHead(channels=16, hidden_channels=8, radius=2)
    features = torch.randn(2, 16, 32)
    centers = features.transpose(1, 2)[:, :5]
    start = torch.tensor([[0.2, 2.5, 4.0, 8.1, 10.0]]).repeat(2, 1)
    end = start + 4.0
    q_start, q_end = head(features, centers, start, end)
    calibrated = calibrate_scores(torch.rand(2, 5, 20), q_start, q_end)
    assert q_start.shape == q_end.shape == (2, 5)
    assert calibrated.shape == (2, 5, 20)
    assert torch.isfinite(calibrated).all()


def test_quality_targets_do_not_backpropagate_to_decoded_boundaries():
    predicted_start = torch.tensor([1.5], requires_grad=True)
    predicted_end = torch.tensor([4.0], requires_grad=True)
    target_start = torch.tensor([1.0])
    target_end = torch.tensor([5.0])
    q_start, q_end = boundary_quality_targets(
        predicted_start, predicted_end, target_start, target_end
    )
    assert not q_start.requires_grad
    assert not q_end.requires_grad

