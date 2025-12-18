"""Tests for data loading and transforms."""

import pytest
import numpy as np
import torch

from genailab.data.loaders import ToyBulkDataset, split_dataset
from genailab.data.transforms import log1p_transform, standardize, hvg_filter
from genailab.data.batch import BatchEncoder, ConditionMetadata
from genailab.data.splits import donor_aware_split, tissue_aware_split


class TestToyBulkDataset:
    def test_dataset_creation(self):
        ds = ToyBulkDataset(n=100, n_genes=50, n_tissues=3, n_diseases=2, n_batches=4)
        assert len(ds) == 100
        assert ds.n_genes == 50

    def test_getitem(self):
        ds = ToyBulkDataset(n=100, n_genes=50)
        item = ds[0]
        assert "x" in item
        assert "cond" in item
        assert item["x"].shape == (50,)
        assert "tissue" in item["cond"]
        assert "disease" in item["cond"]
        assert "batch" in item["cond"]

    def test_split_dataset(self):
        ds = ToyBulkDataset(n=100, n_genes=50)
        train_ds, val_ds = split_dataset(ds, val_frac=0.2)
        assert len(train_ds) == 80
        assert len(val_ds) == 20


class TestTransforms:
    def test_log1p_numpy(self):
        x = np.array([0, 1, 2, 3])
        result = log1p_transform(x)
        expected = np.log1p(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_log1p_torch(self):
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        result = log1p_transform(x)
        expected = torch.log1p(x)
        torch.testing.assert_close(result, expected)

    def test_standardize(self):
        x = np.random.randn(100, 50)
        x_std, mean, std = standardize(x)
        # Should be approximately zero mean, unit variance per gene
        np.testing.assert_array_almost_equal(x_std.mean(axis=0), np.zeros(50), decimal=5)
        np.testing.assert_array_almost_equal(x_std.std(axis=0), np.ones(50), decimal=5)

    def test_hvg_filter(self):
        x = np.random.randn(100, 500)
        x_filtered, idx = hvg_filter(x, n_top_genes=100)
        assert x_filtered.shape == (100, 100)
        assert len(idx) == 100


class TestBatchEncoder:
    def test_fit_transform(self):
        labels = ["A", "B", "A", "C", "B"]
        encoder = BatchEncoder()
        encoded = encoder.fit_transform(labels)
        assert encoded.dtype == np.int64
        assert len(encoded) == 5
        assert encoder.n_categories == 3

    def test_inverse_transform(self):
        labels = ["A", "B", "A", "C", "B"]
        encoder = BatchEncoder()
        encoded = encoder.fit_transform(labels)
        decoded = encoder.inverse_transform(encoded)
        assert decoded == labels


class TestConditionMetadata:
    def test_add_condition(self):
        meta = ConditionMetadata()
        meta.add_condition("tissue", ["brain", "liver", "brain", "heart"])
        meta.add_condition("disease", ["healthy", "sick", "healthy", "sick"])

        assert meta.get_n_categories() == {"tissue": 3, "disease": 2}
        assert len(meta["tissue"]) == 4


class TestSplits:
    def test_donor_aware_split(self):
        n_samples = 100
        # 10 donors, each with 10 samples
        donor_ids = np.repeat(np.arange(10), 10)
        train_idx, val_idx, test_idx = donor_aware_split(n_samples, donor_ids, val_frac=0.2, test_frac=0.1)

        # Check no overlap
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(set(val_idx) & set(test_idx)) == 0

        # Check donors don't leak
        train_donors = set(donor_ids[train_idx])
        val_donors = set(donor_ids[val_idx])
        test_donors = set(donor_ids[test_idx])
        assert len(train_donors & val_donors) == 0
        assert len(train_donors & test_donors) == 0

    def test_tissue_aware_split(self):
        n_samples = 100
        tissue_ids = np.random.randint(0, 5, n_samples)
        train_idx, val_idx, test_idx = tissue_aware_split(n_samples, tissue_ids, stratify=True)

        # Check no overlap
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0
