import math
import types

import torch
import pytest

from models.diffusion_policy import DiffusionPolicy, DiffusionModel, DiffusionConfig


class DummyDiffusionModel(torch.nn.Module):
    """A light-weight stand-in for DiffusionModel used for testing.

    It implements `generate_actions` and `compute_loss` with deterministic simple operations so tests
    can assert expected behaviour without expensive initialization.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def generate_actions(self, batch):
        # batch["observation.state"] has shape (B, n_obs_steps, state_dim)
        B, n_obs, _ = batch["observation.state"].shape
        # return deterministic actions: range values for easy assertions
        total_steps = self.config.n_action_steps
        out = torch.stack([torch.arange(total_steps).float() + b for b in range(B)])
        # shape to (B, n_action_steps, action_dim)
        out = out.unsqueeze(-1).expand(-1, -1, self.config.output_shapes["action"][0])
        return out

    def compute_loss(self, batch):
        # simple MSE to a zero target so loss is just mean square of actions (after normalization)
        actions = batch["action"]
        loss = (actions ** 2).mean()
        return loss


@pytest.fixture(autouse=True)
def stub_diffusion_model(monkeypatch):
    """Replace the heavy DiffusionModel inside DiffusionPolicy with DummyDiffusionModel for tests."""
    monkeypatch.setattr("models.diffusion_policy.DiffusionModel", DummyDiffusionModel)
    yield


def make_config_with_small_image():
    cfg = DiffusionConfig()
    # make small sizes for speed and simplicity
    cfg.input_shapes = {"observation.image": [3, 16, 16], "observation.state": [2]}
    cfg.output_shapes = {"action": [1]}
    cfg.crop_shape = None
    cfg.spatial_softmax_num_keypoints = 4
    cfg.n_obs_steps = 2
    cfg.n_action_steps = 4
    cfg.horizon = 8
    return cfg


def make_dummy_stats(cfg):
    # provide min/max for min_max normalizations used by default config
    stats = {}
    for k, shape in cfg.input_shapes.items():
        if "image" in k:
            stats[k] = {"mean": torch.zeros(shape[0], 1, 1), "std": torch.ones(shape[0], 1, 1)}
        else:
            if cfg.input_normalization_modes.get(k, "min_max") == "min_max":
                stats[k] = {"min": torch.zeros(tuple(shape)), "max": torch.ones(tuple(shape))}
            else:
                stats[k] = {"mean": torch.zeros(tuple(shape)), "std": torch.ones(tuple(shape))}

    for k, shape in cfg.output_shapes.items():
        if cfg.output_normalization_modes.get(k, "min_max") == "min_max":
            stats[k] = {"min": torch.zeros(tuple(shape)), "max": torch.ones(tuple(shape))}
        else:
            stats[k] = {"mean": torch.zeros(tuple(shape)), "std": torch.ones(tuple(shape))}
    return stats


def test_select_action_and_queue_behavior():
    cfg = make_config_with_small_image()
    stats = make_dummy_stats(cfg)

    policy = DiffusionPolicy(cfg, dataset_stats=None)
    policy.eval()

    # Create a batch with batch size 2
    B = 2
    obs_state = torch.zeros((B, cfg.input_shapes["observation.state"][0]))
    img = torch.zeros((B, *cfg.input_shapes["observation.image"]))

    # call select_action repeatedly and assert that actions are produced and queued
    batch = {"observation.state": obs_state, "observation.image": img}

    # First call should populate queues and produce n_action_steps actions inside the queue
    a1 = policy.select_action(batch)
    assert isinstance(a1, torch.Tensor)

    # After first call, the queue should have been filled with n_action_steps - 1 remaining (since one popped)
    remaining = len(policy._queues["action"])
    assert remaining == cfg.n_action_steps - 1

    # Call select_action for more steps and make sure queue cycles through
    actions = [a1]
    for _ in range(remaining + 2):
        actions.append(policy.select_action(batch))

    # Check shapes
    for a in actions:
        assert a.shape == (cfg.output_shapes["action"][0],) or a.dim() == 1


def test_forward_computes_loss():
    cfg = make_config_with_small_image()
    stats = make_dummy_stats(cfg)

    policy = DiffusionPolicy(cfg, dataset_stats=stats)
    policy.train()

    B = 3
    # create inputs shaped as (B, n_obs_steps, ...)
    obs_state = torch.zeros((B, cfg.n_obs_steps, cfg.input_shapes["observation.state"][0]))
    img = torch.zeros((B, cfg.n_obs_steps, 1, *cfg.input_shapes["observation.image"]))

    # create actions shaped (B, horizon, action_dim)
    actions = torch.zeros((B, cfg.horizon, cfg.output_shapes["action"][0]))
    action_is_pad = torch.zeros((B, cfg.horizon), dtype=torch.bool)

    batch = {
        "observation.state": obs_state,
        # DiffusionPolicy expects individual image keys (e.g. "observation.image").
        "observation.image": img,
        "action": actions,
        "action_is_pad": action_is_pad,
    }

    out = policy(batch)
    assert isinstance(out, dict) and "loss" in out
    loss = out["loss"]
    assert torch.is_tensor(loss)
    assert loss.dim() == 0
    # since dummy actions are zeros, and compute_loss uses squared mean of actions, loss should be 0
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
