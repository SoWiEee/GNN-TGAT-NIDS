"""
tests/test_constraints.py
Unit tests for src/attack/constraints.py.

Covers all public methods of ConstraintSet, CoDependencyRule,
SemanticConstraint, and the TCP flag helpers.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.attack.constraints import (
    TCP_ACK,
    TCP_FIN,
    TCP_MAX,
    TCP_PSH,
    TCP_RST,
    TCP_SYN,
    TCP_URG,
    is_valid_tcp_flags,
    nearest_valid_tcp_flags,
    CoDependencyRule,
    ConstraintSet,
    NF_FEATURES,
    SemanticConstraint,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────
# basic_cs and sample_flow are defined in conftest.py (shared with future attack tests).

@pytest.fixture
def feat_idx() -> dict[str, int]:
    return {name: i for i, name in enumerate(NF_FEATURES)}


# ── TCP Flag Helpers ───────────────────────────────────────────────────────────

class TestTCPFlagValidity:
    def test_ack_is_valid(self):
        assert is_valid_tcp_flags(TCP_ACK) is True

    def test_syn_is_valid(self):
        assert is_valid_tcp_flags(TCP_SYN) is True

    def test_syn_ack_is_valid(self):
        assert is_valid_tcp_flags(TCP_SYN | TCP_ACK) is True

    def test_psh_ack_is_valid(self):
        assert is_valid_tcp_flags(TCP_PSH | TCP_ACK) is True

    def test_fin_ack_is_valid(self):
        assert is_valid_tcp_flags(TCP_FIN | TCP_ACK) is True

    def test_null_is_invalid(self):
        assert is_valid_tcp_flags(0x00) is False

    def test_syn_fin_is_invalid(self):
        assert is_valid_tcp_flags(TCP_SYN | TCP_FIN) is False

    def test_syn_rst_is_invalid(self):
        assert is_valid_tcp_flags(TCP_SYN | TCP_RST) is False

    def test_xmas_is_invalid(self):
        assert is_valid_tcp_flags(TCP_MAX) is False

    def test_out_of_range_masked(self):
        # Values > 63 are masked to 6 bits before checking
        assert is_valid_tcp_flags(0x100 | TCP_ACK) is True  # masked → plain ACK


class TestNearestValidTCPFlags:
    def test_valid_input_unchanged(self):
        assert nearest_valid_tcp_flags(TCP_ACK) == TCP_ACK

    def test_syn_fin_projected(self):
        result = nearest_valid_tcp_flags(TCP_SYN | TCP_FIN)
        assert is_valid_tcp_flags(result), f"Projected value {result} is still invalid"

    def test_null_projected(self):
        result = nearest_valid_tcp_flags(0x00)
        assert is_valid_tcp_flags(result)

    def test_xmas_projected(self):
        result = nearest_valid_tcp_flags(TCP_MAX)
        assert is_valid_tcp_flags(result)

    def test_result_in_range(self):
        for flags in range(TCP_MAX + 1):
            result = nearest_valid_tcp_flags(flags)
            assert 0 <= result <= TCP_MAX


# ── CoDependencyRule ───────────────────────────────────────────────────────────

class TestCoDependencyRule:
    def test_recompute_src_bytes(self, sample_flow, feat_idx):
        rule = CoDependencyRule(
            derived="SRC_TO_DST_SECOND_BYTES",
            sources=("IN_BYTES", "FLOW_DURATION_MILLISECONDS"),
            compute=lambda v: v[0] / max(v[1] / 1000.0, 1e-6),
        )
        x = sample_flow.copy()
        x[feat_idx["SRC_TO_DST_SECOND_BYTES"]] = 99999.0   # corrupt
        rule.recompute(x, feat_idx)

        expected = sample_flow[feat_idx["IN_BYTES"]] / (
            sample_flow[feat_idx["FLOW_DURATION_MILLISECONDS"]] / 1000.0
        )
        assert abs(x[feat_idx["SRC_TO_DST_SECOND_BYTES"]] - expected) < 1e-6

    def test_residual_zero_for_consistent(self, sample_flow, feat_idx):
        rule = CoDependencyRule(
            derived="SRC_TO_DST_SECOND_BYTES",
            sources=("IN_BYTES", "FLOW_DURATION_MILLISECONDS"),
            compute=lambda v: v[0] / max(v[1] / 1000.0, 1e-6),
        )
        assert rule.residual(sample_flow, feat_idx) < 1e-6

    def test_residual_nonzero_for_corrupted(self, sample_flow, feat_idx):
        rule = CoDependencyRule(
            derived="SRC_TO_DST_SECOND_BYTES",
            sources=("IN_BYTES", "FLOW_DURATION_MILLISECONDS"),
            compute=lambda v: v[0] / max(v[1] / 1000.0, 1e-6),
        )
        x = sample_flow.copy()
        x[feat_idx["SRC_TO_DST_SECOND_BYTES"]] = 0.0
        assert rule.residual(x, feat_idx) > 0.0


# ── SemanticConstraint ─────────────────────────────────────────────────────────

class TestSemanticConstraint:
    def test_satisfied_within_bounds(self, feat_idx):
        sc = SemanticConstraint(attack_label=2, feature="IN_PKTS", min_value=50.0)
        x = np.zeros(len(NF_FEATURES))
        x[feat_idx["IN_PKTS"]] = 100.0
        assert sc.satisfied(x, feat_idx) is True

    def test_violated_below_min(self, feat_idx):
        sc = SemanticConstraint(attack_label=2, feature="IN_PKTS", min_value=50.0)
        x = np.zeros(len(NF_FEATURES))
        x[feat_idx["IN_PKTS"]] = 10.0
        assert sc.satisfied(x, feat_idx) is False

    def test_project_clips_to_min(self, feat_idx):
        sc = SemanticConstraint(attack_label=2, feature="IN_PKTS", min_value=50.0)
        x = np.zeros(len(NF_FEATURES))
        x[feat_idx["IN_PKTS"]] = 5.0
        sc.project(x, feat_idx)
        assert x[feat_idx["IN_PKTS"]] == 50.0

    def test_project_clips_to_max(self, feat_idx):
        sc = SemanticConstraint(attack_label=3, feature="OUT_BYTES", max_value=5000.0)
        x = np.zeros(len(NF_FEATURES))
        x[feat_idx["OUT_BYTES"]] = 100000.0
        sc.project(x, feat_idx)
        assert x[feat_idx["OUT_BYTES"]] == 5000.0


# ── ConstraintSet ──────────────────────────────────────────────────────────────

class TestConstraintSetProject:
    def test_project_fixes_tcp_flags(self, basic_cs):
        x = np.zeros(len(NF_FEATURES))
        fi = {n: i for i, n in enumerate(NF_FEATURES)}
        x[fi["TCP_FLAGS"]]         = float(TCP_SYN | TCP_FIN)   # invalid
        x[fi["CLIENT_TCP_FLAGS"]]  = float(TCP_PSH | TCP_ACK)   # valid
        x[fi["SERVER_TCP_FLAGS"]]  = float(0x00)                   # invalid

        proj = basic_cs.project(x)
        assert is_valid_tcp_flags(int(proj[fi["TCP_FLAGS"]]))
        assert is_valid_tcp_flags(int(proj[fi["SERVER_TCP_FLAGS"]]))

    def test_project_recomputes_co_dependencies(self, basic_cs, sample_flow):
        fi = {n: i for i, n in enumerate(NF_FEATURES)}
        x = sample_flow.copy()
        x[fi["SRC_TO_DST_SECOND_BYTES"]] = 999999.0   # corrupt

        proj = basic_cs.project(x)
        expected = x[fi["IN_BYTES"]] / (x[fi["FLOW_DURATION_MILLISECONDS"]] / 1000.0)
        assert abs(proj[fi["SRC_TO_DST_SECOND_BYTES"]] - expected) < 1.0

    def test_project_is_idempotent(self, basic_cs, sample_flow):
        p1 = basic_cs.project(sample_flow)
        p2 = basic_cs.project(p1)
        np.testing.assert_array_almost_equal(p1, p2)

    def test_project_batch_shape(self, basic_cs, sample_flow):
        batch = np.stack([sample_flow] * 8)
        result = basic_cs.project(batch)
        assert result.shape == batch.shape

    def test_project_single_equals_batch(self, basic_cs, sample_flow):
        single = basic_cs.project(sample_flow)
        batch = basic_cs.project(sample_flow[np.newaxis, :])
        np.testing.assert_array_almost_equal(single, batch[0])


class TestConstraintSetCheck:
    def test_check_passes_for_valid_flow(self, basic_cs, sample_flow):
        proj = basic_cs.project(sample_flow)
        assert basic_cs.check(proj) is True

    def test_check_fails_invalid_tcp(self, basic_cs, sample_flow):
        fi = {n: i for i, n in enumerate(NF_FEATURES)}
        x = sample_flow.copy()
        x[fi["TCP_FLAGS"]] = float(TCP_SYN | TCP_FIN)
        assert basic_cs.check(x) is False

    def test_check_fails_inconsistent_derived(self, basic_cs, sample_flow):
        fi = {n: i for i, n in enumerate(NF_FEATURES)}
        x = sample_flow.copy()
        x[fi["SRC_TO_DST_SECOND_BYTES"]] = 999999.0
        assert basic_cs.check(x) is False

    def test_check_fails_out_of_bounds(self):
        bounds = {"IN_BYTES": (0.0, 1000.0)}
        cs = ConstraintSet(bounds=bounds, co_dep_rules=[], semantic_constraints=[])
        fi = {n: i for i, n in enumerate(NF_FEATURES)}
        x = np.zeros(len(NF_FEATURES))
        x[fi["TCP_FLAGS"]] = float(TCP_ACK)
        x[fi["CLIENT_TCP_FLAGS"]] = float(TCP_ACK)
        x[fi["SERVER_TCP_FLAGS"]] = float(TCP_ACK)
        x[fi["IN_BYTES"]] = 9999.0   # exceeds bound
        assert cs.check(x) is False


class TestConstraintSetCSR:
    def test_csr_all_valid(self, basic_cs, sample_flow):
        proj = basic_cs.project(sample_flow)
        batch = np.stack([proj] * 10)
        assert basic_cs.csr(batch) == 1.0

    def test_csr_partial(self, basic_cs, sample_flow):
        fi = {n: i for i, n in enumerate(NF_FEATURES)}
        valid = basic_cs.project(sample_flow)

        invalid = sample_flow.copy()
        invalid[fi["TCP_FLAGS"]] = float(TCP_SYN | TCP_FIN)

        # 7 valid, 3 invalid
        batch = np.stack([valid] * 7 + [invalid] * 3)
        csr = basic_cs.csr(batch)
        assert abs(csr - 0.7) < 1e-6

    def test_csr_empty_batch(self, basic_cs):
        assert basic_cs.csr(np.empty((0, len(NF_FEATURES)))) == 1.0


class TestDegreeAnomalyCheck:
    def test_no_anomaly(self, basic_cs):
        orig = np.array([5.0, 6.0, 4.0, 7.0, 5.0])
        new_ = np.array([5.0, 6.0, 4.0, 7.0, 5.0])
        assert basic_cs.check_degree_anomaly(orig, new_) is True

    def test_anomaly_detected(self, basic_cs):
        orig = np.array([2.0, 3.0, 2.0, 3.0, 2.0])
        new_ = np.array([2.0, 3.0, 2.0, 3.0, 1000.0])   # one node explodes
        assert basic_cs.check_degree_anomaly(orig, new_) is False
