import json
import math
from dataclasses import replace

import pytest
import torch

from afloat.diagnostics import capture_inputs
from afloat.formats import FORMATS, FormatPolicy, quantize
from afloat.model import FourierInputs, make_model
from afloat.runner import RunConfig, run_experiment, update_learning_rate
from afloat.targets import detail_component_error, target_values, validation_inputs


def test_fixed_e3m4_uses_its_grid_for_both_weights_and_gradients():
    x = torch.tensor([0.001, 0.031, 0.23, 0.52, 1.125])
    policy = FormatPolicy("fixed-e3m4", {}, 1, 2048)
    for kind in ("weight", "gradient"):
        actual = policy.apply(x, "test", kind, 5)
        assert torch.equal(actual, quantize(x, FORMATS["e3m4"]))
        assert policy.events[-1]["format"] == "e3m4"
    assert torch.equal(policy.evaluate(x, "test"), quantize(x, FORMATS["e3m4"]))


def test_gain_changes_measured_activation_inputs_without_changing_initial_weights():
    base, amplified = make_model(7), make_model(7, 4)
    assert all(
        torch.equal(a, b)
        for a, b in zip(base.parameters(), amplified.parameters(), strict=True)
    )
    x = validation_inputs()[:32]
    _, a = capture_inputs(base, dict(base.named_parameters()), x)
    _, b = capture_inputs(amplified, dict(amplified.named_parameters()), x)
    for family in ("relu", "tanh", "silu"):
        torch.testing.assert_close(
            b[f"1.{family}"], a[f"1.{family}"] * 4, rtol=0, atol=0
        )


def test_default_gain_matches_the_original_forward_and_gradient():
    model = make_model(3)
    x = torch.tensor([[-0.3, 0.4], [0.7, -0.2]])
    h = model[0](x)
    for block in model[1:4]:
        a, b, c = block.expand(h).chunk(3, dim=-1)
        h = h + block.project(
            torch.cat((a.relu(), b.tanh(), torch.nn.functional.silu(c)), dim=-1)
        )
    reference = model[4](h)
    actual = model(x)
    assert torch.equal(actual, reference)
    a = torch.autograd.grad(actual.square().sum(), tuple(model.parameters()))
    b = torch.autograd.grad(reference.square().sum(), tuple(model.parameters()))
    assert all(torch.equal(u, v) for u, v in zip(a, b, strict=True))


def test_input_features_and_component_metric_distinguish_a_missing_feature():
    x = validation_inputs()
    features = FourierInputs()(x)
    assert features.shape == (4225, 18)
    torch.testing.assert_close(
        features[:, 3], torch.sin(2 * math.pi * x[:, 0]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        features[:, 9], torch.sin(8 * math.pi * x[:, 1]), rtol=0, atol=0
    )
    truth = target_values("detail", x)
    assert detail_component_error(truth) == 0
    absent = torch.sin(2 * math.pi * x[:, 0]).unsqueeze(1)
    assert detail_component_error(absent) == pytest.approx(1)
    assert detail_component_error(absent + 0.5 * (truth - absent)) == pytest.approx(
        0.25, rel=1e-5
    )


def test_learning_rate_decay_endpoints_and_old_default():
    config = RunConfig(steps=5, final_learning_rate=0.00001)
    assert update_learning_rate(config, 0) == 0.001
    assert update_learning_rate(config, 4) == 0.00001
    assert [update_learning_rate(config, s) for s in range(5)] == sorted(
        [update_learning_rate(config, s) for s in range(5)], reverse=True
    )
    assert update_learning_rate(replace(config, final_learning_rate=None), 4) == 0.001


def test_followup_checkpoints_replay_and_thinner_logging_preserve_training(tmp_path):
    config = RunConfig(
        targets=("detail",),
        modes=("fixed-e3m4", "adaptive"),
        seeds=(10,),
        steps=4,
        warmup_steps=2,
        batch_size=16,
        eval_every=2,
        adapt_every=2,
        activation_gain=4,
        input_features="fourier",
        final_learning_rate=0.00001,
        format_log_every=3,
        save_checkpoints=True,
    )
    one = run_experiment(config, tmp_path / "one")
    two = run_experiment(replace(config, format_log_every=1), tmp_path / "two")
    for mode in config.modes:
        group = one / "detail/seed-10" / mode
        cp = torch.load(group / "final.pt", weights_only=True)
        other = torch.load(
            two / "detail/seed-10" / mode / "final.pt", weights_only=True
        )
        assert all(torch.equal(v, other["model"][k]) for k, v in cp["model"].items())
        model = make_model(10, **cp["model_settings"])
        model.load_state_dict(cp["model"])
        assert sum(p.numel() for p in model.parameters()) == 10097
        rows = [
            json.loads(s) for s in (group / "metrics.jsonl").read_text().splitlines()
        ]
        assert rows[0]["training_seconds"] == 0
        assert 0 < rows[-1]["training_seconds"] < rows[-1]["elapsed_seconds"]
        assert rows[-1]["gradient_probe_seconds"] > 0
        assert rows[-1]["detail_component_relative_mse"] >= 0
        saved = torch.load(group / "step-4.pt", weights_only=True)
        assert all(torch.equal(v, saved["model"][k]) for k, v in cp["model"].items())
        events = [
            json.loads(s) for s in (group / "formats.jsonl").read_text().splitlines()
        ]
        if mode == "adaptive":
            assert any(
                e["step"] == 2 and e["candidate_mse"] is not None for e in events
            )
    saved_config = json.loads((one / "config.json").read_text())
    assert (
        saved_config["activation_gain"] == 4
        and saved_config["input_features"] == "fourier"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"activation_gain": 0},
        {"activation_gain": float("inf")},
        {"final_learning_rate": 0},
        {"final_learning_rate": 0.1},
        {"input_features": "missing"},
        {"format_log_every": 0},
    ],
)
def test_invalid_followup_settings_fail_before_writing(tmp_path, change):
    with pytest.raises(ValueError):
        run_experiment(replace(RunConfig(), **change), tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
