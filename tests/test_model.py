"""Tests for model components."""

import pytest
import torch

from genailab.model.conditioning import ConditionSpec, ConditionEncoder
from genailab.model.vae import CVAE, CVAE_NB, CVAE_ZINB, elbo_loss
from genailab.objectives.losses import elbo_loss_nb, elbo_loss_zinb, nb_loss, zinb_loss


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


# ============================================================================
# NB and ZINB VAE Tests
# ============================================================================


@pytest.fixture
def count_batch():
    """Sample batch with count data (non-negative integers)."""
    batch_size = 8
    n_genes = 100
    # Simulate count data with Poisson-like distribution
    counts = torch.poisson(torch.ones(batch_size, n_genes) * 10)
    return {
        "x": counts,
        "cond": {
            "tissue": torch.randint(0, 5, (batch_size,)),
            "disease": torch.randint(0, 3, (batch_size,)),
            "batch": torch.randint(0, 8, (batch_size,)),
        },
        "library_size": counts.sum(dim=1),
    }


@pytest.fixture
def cvae_nb(cond_encoder):
    return CVAE_NB(
        n_genes=100,
        z_dim=32,
        cond_encoder=cond_encoder,
        hidden=128,
    )


@pytest.fixture
def cvae_zinb(cond_encoder):
    return CVAE_ZINB(
        n_genes=100,
        z_dim=32,
        cond_encoder=cond_encoder,
        hidden=128,
    )


class TestCVAE_NB:
    def test_encode_shape(self, cvae_nb, count_batch):
        x = count_batch["x"]
        cond = count_batch["cond"]
        mu, logvar = cvae_nb.encode(x, cond)
        assert mu.shape == (8, 32)
        assert logvar.shape == (8, 32)

    def test_decode_shape(self, cvae_nb, count_batch):
        z = torch.randn(8, 32)
        cond = count_batch["cond"]
        out = cvae_nb.decode(z, cond)
        assert out["mu"].shape == (8, 100)
        assert out["theta"].shape == (8, 100)
        assert out["rho"].shape == (8, 100)

    def test_forward_shape(self, cvae_nb, count_batch):
        x = count_batch["x"]
        cond = count_batch["cond"]
        out = cvae_nb(x, cond)
        assert out["mu"].shape == (8, 100)
        assert out["theta"].shape == (8, 100)
        assert out["enc_mu"].shape == (8, 32)
        assert out["enc_logvar"].shape == (8, 32)
        assert out["z"].shape == (8, 32)

    def test_likelihood_attribute(self, cvae_nb):
        assert cvae_nb.likelihood == "nb"


class TestCVAE_ZINB:
    def test_encode_shape(self, cvae_zinb, count_batch):
        x = count_batch["x"]
        cond = count_batch["cond"]
        mu, logvar = cvae_zinb.encode(x, cond)
        assert mu.shape == (8, 32)
        assert logvar.shape == (8, 32)

    def test_decode_shape(self, cvae_zinb, count_batch):
        z = torch.randn(8, 32)
        cond = count_batch["cond"]
        out = cvae_zinb.decode(z, cond)
        assert out["mu"].shape == (8, 100)
        assert out["theta"].shape == (8, 100)
        assert out["pi"].shape == (8, 100)

    def test_forward_shape(self, cvae_zinb, count_batch):
        x = count_batch["x"]
        cond = count_batch["cond"]
        out = cvae_zinb(x, cond)
        assert out["mu"].shape == (8, 100)
        assert out["theta"].shape == (8, 100)
        assert out["pi"].shape == (8, 100)
        assert out["enc_mu"].shape == (8, 32)
        assert out["enc_logvar"].shape == (8, 32)
        assert out["z"].shape == (8, 32)

    def test_likelihood_attribute(self, cvae_zinb):
        assert cvae_zinb.likelihood == "zinb"

    def test_pi_in_valid_range(self, cvae_zinb, count_batch):
        x = count_batch["x"]
        cond = count_batch["cond"]
        out = cvae_zinb(x, cond)
        # pi should be in [0, 1] (sigmoid output)
        assert (out["pi"] >= 0).all()
        assert (out["pi"] <= 1).all()


class TestNBLoss:
    def test_nb_loss_computation(self, cvae_nb, count_batch):
        x = count_batch["x"]
        cond = count_batch["cond"]
        out = cvae_nb(x, cond)
        loss = nb_loss(x, out["mu"], out["theta"])
        assert loss.ndim == 0  # scalar
        assert loss >= 0  # NLL should be non-negative for reasonable inputs

    def test_elbo_loss_nb(self, cvae_nb, count_batch):
        x = count_batch["x"]
        cond = count_batch["cond"]
        out = cvae_nb(x, cond)
        loss, parts = elbo_loss_nb(
            x, out["mu"], out["theta"], out["enc_mu"], out["enc_logvar"]
        )
        assert loss.ndim == 0
        assert "recon" in parts
        assert "kl" in parts
        assert parts["kl"] >= 0

    def test_beta_scaling_nb(self, cvae_nb, count_batch):
        x = count_batch["x"]
        cond = count_batch["cond"]
        out = cvae_nb(x, cond)

        loss_b1, _ = elbo_loss_nb(
            x, out["mu"], out["theta"], out["enc_mu"], out["enc_logvar"], beta=1.0
        )
        loss_b0, parts_b0 = elbo_loss_nb(
            x, out["mu"], out["theta"], out["enc_mu"], out["enc_logvar"], beta=0.0
        )

        # With beta=0, loss should equal reconstruction only
        assert torch.isclose(loss_b0, parts_b0["recon"], atol=1e-5)


class TestZINBLoss:
    def test_zinb_loss_computation(self, cvae_zinb, count_batch):
        x = count_batch["x"]
        cond = count_batch["cond"]
        out = cvae_zinb(x, cond)
        loss = zinb_loss(x, out["mu"], out["theta"], out["pi"])
        assert loss.ndim == 0  # scalar

    def test_elbo_loss_zinb(self, cvae_zinb, count_batch):
        x = count_batch["x"]
        cond = count_batch["cond"]
        out = cvae_zinb(x, cond)
        loss, parts = elbo_loss_zinb(
            x, out["mu"], out["theta"], out["pi"], out["enc_mu"], out["enc_logvar"]
        )
        assert loss.ndim == 0
        assert "recon" in parts
        assert "kl" in parts
        assert parts["kl"] >= 0

    def test_beta_scaling_zinb(self, cvae_zinb, count_batch):
        x = count_batch["x"]
        cond = count_batch["cond"]
        out = cvae_zinb(x, cond)

        loss_b1, _ = elbo_loss_zinb(
            x, out["mu"], out["theta"], out["pi"],
            out["enc_mu"], out["enc_logvar"], beta=1.0
        )
        loss_b0, parts_b0 = elbo_loss_zinb(
            x, out["mu"], out["theta"], out["pi"],
            out["enc_mu"], out["enc_logvar"], beta=0.0
        )

        # With beta=0, loss should equal reconstruction only
        assert torch.isclose(loss_b0, parts_b0["recon"], atol=1e-5)
