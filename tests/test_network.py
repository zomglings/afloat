import math

import pytest
import torch

from afloat.diagnostics import activation_statistics
from afloat.model import make_model
from afloat.targets import TARGETS, feature_metrics, target_values, validation_inputs


def test_model_contains_the_three_activation_families_at_every_depth():
    model = make_model(4)
    assert sum(p.numel() for p in model.parameters()) == 9585
    for family in (torch.nn.ReLU, torch.nn.Tanh, torch.nn.SiLU):
        assert sum(isinstance(m, family) for m in model.modules()) == 3
    x = torch.tensor([[0.1, -0.3], [0.7, 0.2]])
    prediction = model(x)
    assert prediction.shape == (2, 1)
    prediction.square().sum().backward()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("target", TARGETS)
def test_exact_target_has_zero_error_in_all_measured_regions(target):
    x = validation_inputs()
    metrics = feature_metrics(target, x, target_values(target, x))
    assert all(math.isfinite(value) for value in metrics.values())
    for name, value in metrics.items():
        if name != "detail_amplitude":
            assert abs(value) < 1e-7


def test_detail_measurement_detects_a_missing_small_feature():
    x = validation_inputs()
    large_component = torch.sin(2 * math.pi * x[:, 0]).unsqueeze(1)
    metrics = feature_metrics("detail", x, large_component)
    assert metrics["validation_mse"] < 0.0001
    assert metrics["detail_amplitude"] == 0
    assert metrics["detail_amplitude_error"] == pytest.approx(0.01)


def test_activation_diagnostics_compare_identical_parameters_without_error():
    model = make_model(0)
    rows = activation_statistics(
        model, dict(model.named_parameters()), validation_inputs()[:32]
    )
    assert len(rows) == 9
    for row in rows:
        assert row["nonlinear_output_mse"] == 0
        assert 0 <= row["small_slope_fraction"] <= 1
        if row["family"] == "relu":
            assert row["relu_gate_flip_fraction"] == 0
