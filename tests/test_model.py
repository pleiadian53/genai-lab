"""Tests for model components."""

import pytest
import torch

from genailab.model.conditioning import ConditionSpec, ConditionEncoder
from genailab.model.vae import CVAE, elbo_loss


@pytest.fixture
def condition_spec():
    return ConditionSpec(
        n_cats={"tissue": 5, "disease": 3, "batch": 8},
        emb_dim=16,
        out_dim=64,
    )


@pytest.fixture
def cond_encoder(condition_spec):
    return ConditionEncoder(condition_spec)


@pytest.fixture
def cvae(cond_encoder):
    return CVAE(
        n_genes=100,
        z_dim=32,
        cond_encoder=cond_encoder,
        hidden=128,
    )


@pytest.fixture
def sample_batch():
    batch_size = 8
    n_genes = 100
    return {
        "x": torch.randn(batch_size, n_genes),
        "cond": {
            "tissue": torch.randint(0, 5, (batch_size,)),
            "disease": torch.randint(0, 3, (batch_size,)),
            "batch": torch.randint(0, 8, (batch_size,)),
        },
    }


class TestConditionEncoder:
    def test_output_shape(self, cond_encoder, sample_batch):
        cond = sample_batch["cond"]
        out = cond_encoder(cond)
        assert out.shape == (8, 64)

    def test_output_dim_property(self, cond_encoder):
        assert cond_encoder.output_dim == 64


class TestCVAE:
    def test_encode_shape(self, cvae, sample_batch):
        x = sample_batch["x"]
        cond = sample_batch["cond"]
        mu, logvar = cvae.encode(x, cond)
        assert mu.shape == (8, 32)
        assert logvar.shape == (8, 32)

    def test_decode_shape(self, cvae, sample_batch):
        z = torch.randn(8, 32)
        cond = sample_batch["cond"]
        x_hat = cvae.decode(z, cond)
        assert x_hat.shape == (8, 100)

    def test_forward_shape(self, cvae, sample_batch):
        x = sample_batch["x"]
        cond = sample_batch["cond"]
        out = cvae(x, cond)
        assert out["x_hat"].shape == (8, 100)
        assert out["mu"].shape == (8, 32)
        assert out["logvar"].shape == (8, 32)
        assert out["z"].shape == (8, 32)

    def test_sample(self, cvae, sample_batch):
        cond = sample_batch["cond"]
        samples = cvae.sample(8, cond)
        assert samples.shape == (8, 100)


class TestELBOLoss:
    def test_loss_computation(self, cvae, sample_batch):
        x = sample_batch["x"]
        cond = sample_batch["cond"]
        out = cvae(x, cond)
        loss, parts = elbo_loss(x, out["x_hat"], out["mu"], out["logvar"])

        assert loss.ndim == 0  # scalar
        assert "recon" in parts
        assert "kl" in parts
        assert parts["recon"] >= 0
        assert parts["kl"] >= 0

    def test_beta_scaling(self, cvae, sample_batch):
        x = sample_batch["x"]
        cond = sample_batch["cond"]
        out = cvae(x, cond)

        loss_b1, parts_b1 = elbo_loss(x, out["x_hat"], out["mu"], out["logvar"], beta=1.0)
        loss_b0, parts_b0 = elbo_loss(x, out["x_hat"], out["mu"], out["logvar"], beta=0.0)

        # With beta=0, loss should equal reconstruction only
        assert torch.isclose(loss_b0, parts_b0["recon"], atol=1e-5)
