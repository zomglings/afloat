import math

import pytest
import torch

from afloat.formats import FORMATS, FormatPolicy, array_scale, choose_format, quantize


@pytest.mark.parametrize(
    ("name", "dtype", "maximum"),
    [("e4m3", torch.float8_e4m3fn, 448.0), ("e5m2", torch.float8_e5m2, 57344.0)],
)
def test_standard_formats_agree_with_native_pytorch(name, dtype, maximum):
    positive = torch.logspace(-8, 5, 10001).clamp_max(maximum)
    x = torch.cat((-positive, torch.zeros(1), positive))
    expected = x.to(dtype).float()
    torch.testing.assert_close(
        quantize(x, FORMATS[name], scale=1), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    ("name", "values", "expected"),
    [
        (
            "e4m3",
            [1.0625, 1.1875, -1.1875, 0.0009765625, 900],
            [1, 1.25, -1.25, 0, 448],
        ),
        (
            "e5m2",
            [1.125, 1.375, -1.375, 0.00000762939453125, 90000],
            [1, 1.5, -1.5, 0, 57344],
        ),
        (
            "e3m4",
            [1.03125, 1.09375, -1.09375, 0.0078125, 99],
            [1, 1.125, -1.125, 0, 15.5],
        ),
    ],
)
def test_ties_subnormals_sign_and_saturation(name, values, expected):
    actual = quantize(torch.tensor(values), FORMATS[name], scale=1)
    torch.testing.assert_close(actual, torch.tensor(expected), rtol=0, atol=0)


def test_zero_arrays_and_smallest_fp32_value():
    x = torch.zeros(5)
    assert array_scale(x, FORMATS["e4m3"]) == 1
    assert torch.equal(quantize(x, FORMATS["e4m3"]), x)
    tiny = torch.tensor([math.ldexp(1.0, -149)])
    assert torch.equal(quantize(tiny, FORMATS["e4m3"]), tiny)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_are_rejected(value):
    with pytest.raises(FloatingPointError, match="finite"):
        quantize(torch.tensor([value]), FORMATS["e4m3"])


def test_simulation_overflow_fails_visibly():
    with pytest.raises(FloatingPointError, match="simulation range"):
        quantize(torch.tensor([torch.finfo(torch.float32).max]), FORMATS["e4m3"])


def test_selection_tracks_distribution_shape_and_calibration_stays_fixed():
    narrow = torch.linspace(0.1, 1, 2000)
    wide = torch.cat((torch.linspace(0.000001, 0.00001, 2000), torch.ones(1)))
    narrow_choice, wide_choice = choose_format(narrow), choose_format(wide)
    assert narrow_choice == "e3m4"
    assert wide_choice == "e5m2"
    choices = {"weight:test": narrow_choice}
    adaptive = FormatPolicy("adaptive", choices, interval=2, sample_size=2048)
    calibrated = FormatPolicy("calibrated", choices, interval=2, sample_size=2048)
    adaptive.apply(wide, "test", "weight", 1)
    assert adaptive.choices["weight:test"] == narrow_choice
    adaptive.apply(wide, "test", "weight", 2)
    calibrated.apply(wide, "test", "weight", 2)
    assert adaptive.choices["weight:test"] == wide_choice
    assert calibrated.choices["weight:test"] == narrow_choice
    assert choices["weight:test"] == narrow_choice
    switched = adaptive.events[-1]
    old = quantize(wide, FORMATS[narrow_choice])
    new = quantize(wide, FORMATS[wide_choice])
    assert switched["changed_elements"] == int((old != new).sum()) > 0
    assert switched["rounding_change_mse"] == float(
        (old.double() - new.double()).square().mean()
    )


def test_fixed_and_customized_formats_share_identical_scaling():
    choices = {"weight:test": "e4m3"}
    fixed = FormatPolicy("fixed-e4m3", choices, 2, 2048)
    customized = FormatPolicy("calibrated", choices, 2, 2048)
    for step, magnitude in enumerate([0.0001, 1, 1000]):
        x = torch.linspace(-magnitude, magnitude, 100)
        assert torch.equal(
            fixed.apply(x, "test", "weight", step),
            customized.apply(x, "test", "weight", step),
        )
        assert fixed.events[-1]["scale"] == customized.events[-1]["scale"]
    assert len({event["scale"] for event in fixed.events}) == 3


def test_evaluation_does_not_reselect_or_log():
    policy = FormatPolicy("adaptive", {"weight:test": "e4m3"}, 1, 2048)
    policy.evaluate(torch.linspace(0.1, 1, 100), "test")
    assert policy.choices == {"weight:test": "e4m3"}
    assert not policy.events


def test_rounding_has_identity_gradient_only_when_requested():
    w = torch.tensor([1.0625], requires_grad=True)
    q = quantize(w, FORMATS["e4m3"], scale=1)
    assert not q.requires_grad
    used = w + (q - w).detach()
    used.square().sum().backward()
    torch.testing.assert_close(w.grad, torch.tensor([2.0]))
