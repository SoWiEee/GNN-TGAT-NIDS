"""Tests for PCAP to NetFlow conversion (unit tests only — no nfstream dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.pcap_to_netflow import TCP_ACK, TCP_PSH, TCP_SYN, _build_tcp_flags, _compute_derived


class TestBuildTCPFlags:
    def test_syn_flag(self):
        row = pd.Series({"_syn_src": 1, "_fin_src": 0, "_rst_src": 0, "_psh_src": 0, "_ack_src": 0})
        assert _build_tcp_flags(row, "src") == TCP_SYN

    def test_ack_flag(self):
        row = pd.Series({"_syn_src": 0, "_fin_src": 0, "_rst_src": 0, "_psh_src": 0, "_ack_src": 1})
        assert _build_tcp_flags(row, "src") == TCP_ACK

    def test_combined_flags(self):
        row = pd.Series({"_syn_src": 1, "_fin_src": 0, "_rst_src": 0, "_psh_src": 1, "_ack_src": 1})
        assert _build_tcp_flags(row, "src") == (TCP_SYN | TCP_PSH | TCP_ACK)

    def test_no_flags(self):
        row = pd.Series({"_syn_src": 0, "_fin_src": 0, "_rst_src": 0, "_psh_src": 0, "_ack_src": 0})
        assert _build_tcp_flags(row, "src") == 0

    def test_dst_prefix(self):
        row = pd.Series({"_syn_dst": 0, "_fin_dst": 0, "_rst_dst": 0, "_psh_dst": 0, "_ack_dst": 1})
        assert _build_tcp_flags(row, "dst") == TCP_ACK


class TestComputeDerived:
    def test_throughput_computed(self):
        df = pd.DataFrame({
            "IN_BYTES": [2000.0],
            "OUT_BYTES": [800.0],
            "FLOW_DURATION_MILLISECONDS": [4000.0],
        })
        result = _compute_derived(df)
        assert "SRC_TO_DST_SECOND_BYTES" in result.columns
        assert "DST_TO_SRC_SECOND_BYTES" in result.columns
        assert "SRC_TO_DST_AVG_THROUGHPUT" in result.columns
        assert "DST_TO_SRC_AVG_THROUGHPUT" in result.columns

        assert abs(result["SRC_TO_DST_SECOND_BYTES"].iloc[0] - 500.0) < 1e-6
        assert abs(result["DST_TO_SRC_SECOND_BYTES"].iloc[0] - 200.0) < 1e-6

    def test_zero_duration_clamped(self):
        df = pd.DataFrame({
            "IN_BYTES": [1000.0],
            "OUT_BYTES": [500.0],
            "FLOW_DURATION_MILLISECONDS": [0.0],
        })
        result = _compute_derived(df)
        assert np.isfinite(result["SRC_TO_DST_SECOND_BYTES"].iloc[0])
        assert np.isfinite(result["SRC_TO_DST_AVG_THROUGHPUT"].iloc[0])
