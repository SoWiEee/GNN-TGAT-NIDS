"""Tests for static_builder.py and static_dataset.py."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.static_builder import build_static_graphs
from src.data.static_dataset import StaticNIDSDataset


# ── Helpers ───────────────────────────────────────────────────────────────────

_CSV_TEMPLATE = textwrap.dedent(
    """\
    FIRST_SEEN,IPV4_SRC_ADDR,L4_SRC_PORT,IPV4_DST_ADDR,L4_DST_PORT,Label,IN_BYTES,OUT_BYTES,IN_PKTS,OUT_PKTS,FLOW_DURATION_MILLISECONDS,TCP_FLAGS,PROTOCOL
    2023-01-01 00:00:01,10.0.0.1,1234,10.0.0.2,80,Benign,100,200,3,4,500,2,6
    2023-01-01 00:00:10,10.0.0.3,5678,10.0.0.4,443,Benign,150,250,5,6,600,16,6
    2023-01-01 00:00:20,10.0.0.5,9012,10.0.0.6,22,DoS,2000,500,50,10,100,2,6
    2023-01-01 00:01:05,10.0.0.7,3456,10.0.0.8,80,Benign,80,160,2,3,400,18,6
    2023-01-01 00:01:15,10.0.0.1,1234,10.0.0.9,443,Recon,50,50,1,1,200,2,6
    2023-01-01 00:01:25,10.0.0.2,4321,10.0.0.3,80,Benign,120,240,4,5,550,16,6
    2023-01-01 00:02:00,10.0.0.4,6789,10.0.0.5,22,DoS,3000,600,60,12,150,2,6
    2023-01-01 00:02:30,10.0.0.6,2345,10.0.0.7,443,Benign,90,180,3,4,450,18,6
    2023-01-01 00:03:00,10.0.0.8,7890,10.0.0.1,80,Recon,60,60,2,2,300,2,6
    2023-01-01 00:03:30,10.0.0.9,3456,10.0.0.2,22,Benign,110,220,4,5,500,16,6
    """
)


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    p = tmp_path / "sample.csv"
    p.write_text(_CSV_TEMPLATE)
    return p


@pytest.fixture
def built_dir(sample_csv: Path, tmp_path: Path) -> Path:
    out = tmp_path / "processed"
    build_static_graphs(
        csv_path=sample_csv,
        output_dir=out,
        window_size_s=60.0,
        ratios=(0.6, 0.2, 0.2),
        clip_sigma=3.0,
    )
    return out


# ── build_static_graphs ───────────────────────────────────────────────────────

class TestBuildStaticGraphs:
    def test_output_directories_created(self, built_dir: Path):
        assert (built_dir / "train").is_dir()
        assert (built_dir / "val").is_dir()
        assert (built_dir / "test").is_dir()

    def test_meta_json_exists(self, built_dir: Path):
        assert (built_dir / "meta.json").exists()

    def test_scaler_pkl_exists(self, built_dir: Path):
        assert (built_dir / "scaler.pkl").exists()

    def test_label2idx_json_exists(self, built_dir: Path):
        assert (built_dir / "label2idx.json").exists()

    def test_meta_has_required_keys(self, built_dir: Path):
        with open(built_dir / "meta.json") as f:
            meta = json.load(f)
        for key in ("label2idx", "n_classes", "n_features", "split_counts", "window_size_s"):
            assert key in meta, f"Missing key: {key}"

    def test_label2idx_benign_is_zero(self, built_dir: Path):
        with open(built_dir / "label2idx.json") as f:
            label2idx = json.load(f)
        assert label2idx.get("Benign") == 0

    def test_graphs_saved_as_pt_files(self, built_dir: Path):
        all_pts = list(built_dir.rglob("*.pt"))
        assert len(all_pts) > 0, "No .pt files found"

    def test_graph_has_correct_attributes(self, built_dir: Path):
        pt_files = sorted((built_dir / "train").glob("*.pt"))
        if not pt_files:
            pytest.skip("No train graphs produced (depends on window/split sizes)")
        data = torch.load(pt_files[0], weights_only=False)
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "edge_attr")
        assert hasattr(data, "y")
        assert hasattr(data, "y_multi")

    def test_edge_index_valid_node_range(self, built_dir: Path):
        for split in ("train", "val", "test"):
            for pt_file in (built_dir / split).glob("*.pt"):
                data = torch.load(pt_file, weights_only=False)
                assert data.edge_index.max() < data.num_nodes

    def test_split_counts_sum_to_total_graphs(self, built_dir: Path):
        with open(built_dir / "meta.json") as f:
            meta = json.load(f)
        counts = meta["split_counts"]
        total = sum(counts.values())
        all_pts = list(built_dir.rglob("*.pt"))
        assert total == len(all_pts)


# ── StaticNIDSDataset ─────────────────────────────────────────────────────────

class TestStaticNIDSDataset:
    def test_len_matches_file_count(self, built_dir: Path):
        for split in ("train", "val", "test"):
            ds = StaticNIDSDataset(built_dir, split=split)
            pt_count = len(list((built_dir / split).glob("*.pt")))
            assert len(ds) == pt_count

    def test_get_returns_data_object(self, built_dir: Path):
        from torch_geometric.data import Data

        for split in ("train", "val", "test"):
            ds = StaticNIDSDataset(built_dir, split=split)
            if len(ds) == 0:
                continue
            item = ds.get(0)
            assert isinstance(item, Data)

    def test_n_classes_from_meta(self, built_dir: Path):
        ds = StaticNIDSDataset(built_dir, split="train")
        with open(built_dir / "meta.json") as f:
            meta = json.load(f)
        assert ds.n_classes == meta["n_classes"]

    def test_invalid_split_raises(self, built_dir: Path):
        with pytest.raises(ValueError, match="split must be"):
            StaticNIDSDataset(built_dir, split="holdout")

    def test_missing_meta_raises(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="meta.json"):
            StaticNIDSDataset(empty_dir, split="train")

    def test_scaler_loadable(self, built_dir: Path):
        from sklearn.preprocessing import StandardScaler

        ds = StaticNIDSDataset(built_dir, split="train")
        scaler = ds.scaler
        assert isinstance(scaler, StandardScaler)

    def test_label2idx_loadable(self, built_dir: Path):
        ds = StaticNIDSDataset(built_dir, split="train")
        label2idx = ds.label2idx
        assert isinstance(label2idx, dict)
        assert "Benign" in label2idx
