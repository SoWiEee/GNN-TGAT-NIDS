"""Shared pytest fixtures for the GARF-NIDS test suite."""

from __future__ import annotations

import numpy as np
import pytest

from src.attack.constraints import (
    TCP_ACK,
    TCP_PSH,
    ConstraintSet,
    NF_FEATURES,
)


@pytest.fixture
def basic_cs() -> ConstraintSet:
    """ConstraintSet with wide bounds (effectively unconstrained on bounds)."""
    bounds = {name: (0.0, 1e9) for name in NF_FEATURES}
    return ConstraintSet(bounds=bounds)


@pytest.fixture
def sample_flow() -> np.ndarray:
    """A physically plausible raw-scale NetFlow feature vector.

    Derived features (throughput, byte-rate) are set to be algebraically
    consistent with IN_BYTES, OUT_BYTES, and FLOW_DURATION_MILLISECONDS so
    that constraint-check tests start from a valid baseline.
    """
    x = np.zeros(len(NF_FEATURES), dtype=np.float64)
    fi = {name: i for i, name in enumerate(NF_FEATURES)}

    x[fi["IN_BYTES"]]                   = 2000.0
    x[fi["OUT_BYTES"]]                  = 800.0
    x[fi["IN_PKTS"]]                    = 20.0
    x[fi["OUT_PKTS"]]                   = 10.0
    x[fi["FLOW_DURATION_MILLISECONDS"]] = 4000.0   # 4 seconds
    x[fi["TCP_FLAGS"]]                  = float(TCP_ACK)
    x[fi["CLIENT_TCP_FLAGS"]]           = float(TCP_PSH | TCP_ACK)
    x[fi["SERVER_TCP_FLAGS"]]           = float(TCP_ACK)

    flow_s = x[fi["FLOW_DURATION_MILLISECONDS"]] / 1000.0
    x[fi["SRC_TO_DST_SECOND_BYTES"]]   = x[fi["IN_BYTES"]]  / flow_s
    x[fi["DST_TO_SRC_SECOND_BYTES"]]   = x[fi["OUT_BYTES"]] / flow_s
    x[fi["SRC_TO_DST_AVG_THROUGHPUT"]] = x[fi["IN_BYTES"]]  * 8.0 / flow_s
    x[fi["DST_TO_SRC_AVG_THROUGHPUT"]] = x[fi["OUT_BYTES"]] * 8.0 / flow_s

    return x
