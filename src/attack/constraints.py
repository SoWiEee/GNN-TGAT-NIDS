"""
src/attack/constraints.py
Constraint set for CAAG adversarial example generation (NF-UNSW-NB15-v2).

Operates on **raw** (inverse-transformed) feature vectors. The attack module
is responsible for:
  1. Calling scaler.inverse_transform() before project() / check().
  2. Calling scaler.transform() afterwards to re-enter normalised space.

Usage:
    cs = ConstraintSet.from_scaler("data/processed/static/scaler.pkl")

    # inside each PGD step:
    x_raw = scaler.inverse_transform(x_adv)
    x_raw = cs.project(x_raw, attack_label=2)
    x_adv = scaler.transform(x_raw)

    # final gate (CSR = 1.0 required):
    assert cs.csr(x_raw_batch) == 1.0
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

# ── Feature Schema ─────────────────────────────────────────────────────────────
# NF-UNSW-NB15-v2 — 34 numeric features (IP addresses excluded; label columns
# excluded). Update this list if the dataset preprocessing changes column order.

NF_FEATURES: list[str] = [
    "L4_SRC_PORT",                  #  0
    "L4_DST_PORT",                  #  1
    "PROTOCOL",                     #  2
    "L7_PROTO",                     #  3
    "IN_BYTES",                     #  4
    "OUT_BYTES",                    #  5
    "IN_PKTS",                      #  6
    "OUT_PKTS",                     #  7
    "TCP_FLAGS",                    #  8
    "CLIENT_TCP_FLAGS",             #  9
    "SERVER_TCP_FLAGS",             # 10
    "FLOW_DURATION_MILLISECONDS",   # 11
    "DURATION_IN",                  # 12
    "DURATION_OUT",                 # 13
    "MIN_TTL",                      # 14
    "MAX_TTL",                      # 15
    "LONGEST_FLOW_PKT",             # 16
    "SHORTEST_FLOW_PKT",            # 17
    "MIN_IP_PKT_LEN",               # 18
    "MAX_IP_PKT_LEN",               # 19
    "SRC_TO_DST_SECOND_BYTES",      # 20
    "DST_TO_SRC_SECOND_BYTES",      # 21
    "RETRANSMITTED_IN_BYTES",       # 22
    "RETRANSMITTED_OUT_BYTES",      # 23
    "RETRANSMITTED_IN_PKTS",        # 24
    "RETRANSMITTED_OUT_PKTS",       # 25
    "SRC_TO_DST_AVG_THROUGHPUT",    # 26
    "DST_TO_SRC_AVG_THROUGHPUT",    # 27
    "NUM_PKTS_UP_TO_128_BYTES",     # 28
    "NUM_PKTS_128_TO_256_BYTES",    # 29
    "NUM_PKTS_256_TO_512_BYTES",    # 30
    "NUM_PKTS_512_TO_1024_BYTES",   # 31
    "NUM_PKTS_1024_TO_1514_BYTES",  # 32
    "TCP_WIN_MAX_IN",               # 33
]

_FEAT_IDX: dict[str, int] = {name: i for i, name in enumerate(NF_FEATURES)}

# ── TCP Flag Validity ──────────────────────────────────────────────────────────
# 6-bit bitmask: bit5=URG  bit4=ACK  bit3=PSH  bit2=RST  bit1=SYN  bit0=FIN

TCP_FIN: int = 0x01
TCP_SYN: int = 0x02
TCP_RST: int = 0x04
TCP_PSH: int = 0x08
TCP_ACK: int = 0x10
TCP_URG: int = 0x20
TCP_MAX: int = 0x3F

# RFC 793 + common security heuristics — these combinations are invalid
_INVALID_TCP_COMBOS: frozenset[int] = frozenset({
    0x00,                               # NULL: no flags set
    TCP_SYN | TCP_FIN,                 # 0x03 — contradictory state signals
    TCP_SYN | TCP_RST,                 # 0x06 — contradictory
    TCP_SYN | TCP_FIN | TCP_RST,       # 0x07
    TCP_MAX,                            # 0x3F — XMAS scan
})

_TCP_FLAG_COLS: tuple[str, ...] = (
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
)

_EPS: float = 1e-6  # guard against division by zero


def is_valid_tcp_flags(flags: int) -> bool:
    """Return True if the TCP bitmask is a protocol-valid combination.

    Args:
        flags: Integer TCP flags value (6-bit, 0–63).

    Returns:
        True if flags are valid per RFC 793 heuristics.
    """
    flags = int(flags) & TCP_MAX
    return flags not in _INVALID_TCP_COMBOS


def nearest_valid_tcp_flags(flags: int) -> int:
    """Project an invalid TCP flag value to the nearest valid one.

    Strategy: remove bits in priority order (FIN first, then SYN) until the
    result is no longer in the invalid set. Falls back to plain ACK if all
    single-bit removals fail.

    Args:
        flags: Possibly-invalid TCP flags integer.

    Returns:
        A valid TCP flags integer.
    """
    flags = int(flags) & TCP_MAX
    if flags not in _INVALID_TCP_COMBOS:
        return flags
    for bit in (TCP_FIN, TCP_SYN, TCP_RST, TCP_URG):
        candidate = flags & ~bit
        if candidate not in _INVALID_TCP_COMBOS:
            return candidate
    return TCP_ACK  # safe fallback


# ── Feature Co-Dependency Rules ────────────────────────────────────────────────

@dataclass
class CoDependencyRule:
    """A derived feature and how to recompute it from source features.

    Args:
        derived: Name of the derived (dependent) feature.
        sources: Names of the source features used in the computation.
        compute: Function that takes a list of source values and returns the
            expected derived value.
    """

    derived: str
    sources: tuple[str, ...]
    compute: Callable[[list[float]], float]

    def recompute(self, x: np.ndarray, feat_idx: dict[str, int]) -> None:
        """Overwrite the derived feature in ``x`` with its recomputed value.

        Args:
            x: 1-D raw-scale feature array (modified in place).
            feat_idx: Mapping from feature name to array index.
        """
        src_vals = [float(x[feat_idx[s]]) for s in self.sources]
        x[feat_idx[self.derived]] = self.compute(src_vals)

    def residual(self, x: np.ndarray, feat_idx: dict[str, int]) -> float:
        """Absolute difference between stored and recomputed derived value.

        Args:
            x: 1-D raw-scale feature array.
            feat_idx: Mapping from feature name to array index.

        Returns:
            |actual − expected|. 0.0 means perfectly consistent.
        """
        src_vals = [float(x[feat_idx[s]]) for s in self.sources]
        expected = self.compute(src_vals)
        actual = float(x[feat_idx[self.derived]])
        return abs(actual - expected)


def _default_co_dep_rules() -> list[CoDependencyRule]:
    """Return the default co-dependency rules for NF-UNSW-NB15-v2.

    All byte-rate and throughput features are derived from byte counts and
    flow duration. They must be recomputed after any perturbation that touches
    IN_BYTES, OUT_BYTES, or FLOW_DURATION_MILLISECONDS.
    """
    return [
        CoDependencyRule(
            derived="SRC_TO_DST_SECOND_BYTES",
            sources=("IN_BYTES", "FLOW_DURATION_MILLISECONDS"),
            compute=lambda v: v[0] / max(v[1] / 1000.0, _EPS),
        ),
        CoDependencyRule(
            derived="DST_TO_SRC_SECOND_BYTES",
            sources=("OUT_BYTES", "FLOW_DURATION_MILLISECONDS"),
            compute=lambda v: v[0] / max(v[1] / 1000.0, _EPS),
        ),
        CoDependencyRule(
            derived="SRC_TO_DST_AVG_THROUGHPUT",
            sources=("IN_BYTES", "FLOW_DURATION_MILLISECONDS"),
            compute=lambda v: (v[0] * 8.0) / max(v[1] / 1000.0, _EPS),
        ),
        CoDependencyRule(
            derived="DST_TO_SRC_AVG_THROUGHPUT",
            sources=("OUT_BYTES", "FLOW_DURATION_MILLISECONDS"),
            compute=lambda v: (v[0] * 8.0) / max(v[1] / 1000.0, _EPS),
        ),
    ]


# ── Semantic Preservation ──────────────────────────────────────────────────────

@dataclass
class SemanticConstraint:
    """Per-attack-type invariant that must hold after perturbation.

    Example: a DDoS flow must maintain high packet rate; perturbing it to
    appear like a low-volume benign flow would change its attack semantics.

    Args:
        attack_label: Integer label from NF-UNSW-NB15-v2 (0=Benign, 1=DoS,
            2=DDoS, 3=Reconnaissance, …). Update the label mapping after EDA.
        feature: Feature name to constrain.
        min_value: Lower bound (inclusive). ``None`` means no lower bound.
        max_value: Upper bound (inclusive). ``None`` means no upper bound.
    """

    attack_label: int
    feature: str
    min_value: float | None = None
    max_value: float | None = None

    def satisfied(self, x: np.ndarray, feat_idx: dict[str, int]) -> bool:
        """Return True if the constraint holds for the given feature vector.

        Args:
            x: 1-D raw-scale feature array.
            feat_idx: Feature name → index mapping.

        Returns:
            True if feature value is within [min_value, max_value].
        """
        val = float(x[feat_idx[self.feature]])
        if self.min_value is not None and val < self.min_value:
            return False
        if self.max_value is not None and val > self.max_value:
            return False
        return True

    def project(self, x: np.ndarray, feat_idx: dict[str, int]) -> None:
        """Clip the feature to satisfy the constraint in place.

        Args:
            x: 1-D raw-scale feature array (modified in place).
            feat_idx: Feature name → index mapping.
        """
        idx = feat_idx[self.feature]
        if self.min_value is not None:
            x[idx] = max(float(x[idx]), self.min_value)
        if self.max_value is not None:
            x[idx] = min(float(x[idx]), self.max_value)


def _default_semantic_constraints() -> list[SemanticConstraint]:
    """Return example semantic constraints for NF-UNSW-NB15-v2.

    Values are conservative placeholders. Update ``min_value`` / ``max_value``
    with percentile statistics from the training split after EDA.

    Attack label mapping (verify against actual dataset):
        0 = Benign
        1 = DoS
        2 = DDoS
        3 = Reconnaissance
        4–8 = Other attack categories
    """
    return [
        # DDoS must maintain high inbound packet count
        SemanticConstraint(attack_label=2, feature="IN_PKTS", min_value=50.0),
        # DoS must retain substantial inbound bytes
        SemanticConstraint(attack_label=1, feature="IN_BYTES", min_value=100.0),
        # Reconnaissance typically has low byte volume — cap outbound
        SemanticConstraint(attack_label=3, feature="OUT_BYTES", max_value=5000.0),
    ]


# ── ConstraintSet ──────────────────────────────────────────────────────────────

class ConstraintSet:
    """Unified constraint set for CAAG adversarial example generation.

    Enforces five constraint categories (see spec.md §3.3.3):
    1. Feature bounds — per-feature empirical [min, max] from training data.
    2. Feature co-dependency — derived features recomputed algebraically.
    3. TCP flag validity — RFC 793 protocol state machine compliance.
    4. Semantic preservation — per-attack-type invariants.
    5. Degree anomaly limit — for edge-injection attacks (checked separately).

    Args:
        feature_names: Ordered list of feature column names matching the
            dataset preprocessing output. Defaults to ``NF_FEATURES``.
        bounds: Dict mapping feature name → ``(min, max)`` from training data.
            Features not in this dict are unconstrained.
        co_dep_rules: Co-dependency recomputation rules. Defaults to the four
            byte-rate / throughput rules for NF-UNSW-NB15-v2.
        semantic_constraints: Per-attack-type invariants. Defaults to three
            placeholder constraints — update after EDA on the training split.
        tcp_flag_cols: Names of TCP flag columns. Subset of ``feature_names``.
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        bounds: dict[str, tuple[float, float]] | None = None,
        co_dep_rules: list[CoDependencyRule] | None = None,
        semantic_constraints: list[SemanticConstraint] | None = None,
        tcp_flag_cols: tuple[str, ...] = _TCP_FLAG_COLS,
    ) -> None:
        self.feature_names: list[str] = feature_names if feature_names is not None else NF_FEATURES
        self._feat_idx: dict[str, int] = {n: i for i, n in enumerate(self.feature_names)}
        self.bounds: dict[str, tuple[float, float]] = bounds or {}
        self.co_dep_rules: list[CoDependencyRule] = (
            co_dep_rules if co_dep_rules is not None else _default_co_dep_rules()
        )
        self.semantic_constraints: list[SemanticConstraint] = (
            semantic_constraints if semantic_constraints is not None
            else _default_semantic_constraints()
        )
        # Only keep columns that actually exist in feature_names
        self.tcp_flag_cols: list[str] = [c for c in tcp_flag_cols if c in self._feat_idx]

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_scaler(
        cls,
        scaler_path: str | Path,
        feature_names: list[str] | None = None,
        **kwargs,
    ) -> "ConstraintSet":
        """Build a ConstraintSet with feature bounds derived from a fitted scaler.

        Bounds are set to ``mean ± 3σ`` from the StandardScaler, clipped to
        ``[0, ∞)`` for non-negative features to prevent physically impossible
        negative byte/packet counts.

        Args:
            scaler_path: Path to a pickled ``sklearn.preprocessing.StandardScaler``
                saved by ``src/data/static_builder.py`` or
                ``src/data/temporal_builder.py``.
            feature_names: Feature column list. Defaults to ``NF_FEATURES``.
            **kwargs: Forwarded to ``__init__`` (e.g. ``semantic_constraints``).

        Returns:
            A fully configured ``ConstraintSet`` instance.
        """
        feature_names = feature_names or NF_FEATURES
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        n_scaler = len(scaler.mean_)
        n_names = len(feature_names)
        if n_scaler != n_names:
            raise ValueError(
                f"Scaler was fitted on {n_scaler} features but feature_names "
                f"contains {n_names} entries. Rebuild the scaler or pass the "
                f"correct feature_names list."
            )

        bounds: dict[str, tuple[float, float]] = {}
        for name, mean, std in zip(feature_names, scaler.mean_, scaler.scale_):
            # Lower bound floored at 0 for features with natural non-negative range
            # (byte/packet counts, ports, durations). Features that can legitimately
            # be negative (e.g. deltas) would need a per-feature allow-list here.
            lo = max(0.0, float(mean) - 3.0 * float(std))
            hi = float(mean) + 3.0 * float(std)
            bounds[name] = (lo, hi)

        return cls(feature_names=feature_names, bounds=bounds, **kwargs)

    # ── Project ───────────────────────────────────────────────────────────

    def project(
        self,
        x_raw: np.ndarray,
        attack_label: int | None = None,
    ) -> np.ndarray:
        """Project raw feature vector(s) to the constraint-satisfying region.

        Projection order (matches spec.md sequence diagram):
        bounds → co-dependency recomputation → TCP flags → semantic preservation.

        Args:
            x_raw: Shape ``(n_features,)`` or ``(batch, n_features)``.
                Must be in raw (non-normalised) scale.
            attack_label: Optional attack class integer for semantic constraints.
                Pass ``None`` to skip semantic enforcement.

        Returns:
            Projected array with the same shape as ``x_raw``.
        """
        x = np.array(x_raw, dtype=np.float64)
        if x.ndim == 1:
            return self._project_single(x, attack_label)
        for i in range(len(x)):
            x[i] = self._project_single(x[i], attack_label)
        return x

    def _project_single(self, x: np.ndarray, attack_label: int | None) -> np.ndarray:
        x = self._clip_bounds(x)
        x = self._fix_co_dependencies(x)
        x = self._fix_tcp_flags(x)
        if attack_label is not None:
            x = self._fix_semantic(x, attack_label)
        return x

    def _clip_bounds(self, x: np.ndarray) -> np.ndarray:
        for name, (lo, hi) in self.bounds.items():
            if name in self._feat_idx:
                x[self._feat_idx[name]] = np.clip(x[self._feat_idx[name]], lo, hi)
        return x

    def _fix_co_dependencies(self, x: np.ndarray) -> np.ndarray:
        for rule in self.co_dep_rules:
            if rule.derived in self._feat_idx:
                rule.recompute(x, self._feat_idx)
        return x

    def _fix_tcp_flags(self, x: np.ndarray) -> np.ndarray:
        for col in self.tcp_flag_cols:
            idx = self._feat_idx[col]
            flags = int(round(float(x[idx]))) & TCP_MAX
            x[idx] = float(nearest_valid_tcp_flags(flags))
        return x

    def _fix_semantic(self, x: np.ndarray, attack_label: int) -> np.ndarray:
        for sc in self.semantic_constraints:
            if sc.attack_label == attack_label and sc.feature in self._feat_idx:
                sc.project(x, self._feat_idx)
        return x

    # ── Check ─────────────────────────────────────────────────────────────

    def check(
        self,
        x_raw: np.ndarray,
        attack_label: int | None = None,
        rtol: float = 0.01,
    ) -> bool:
        """Return True only if **all** active constraints are satisfied.

        Args:
            x_raw: 1-D raw-scale feature vector.
            attack_label: Optional attack class for semantic checks.
            rtol: Relative tolerance for co-dependency checks (default 1%).

        Returns:
            True if every constraint passes; False on the first violation.
        """
        if not self._check_bounds(x_raw):
            return False
        if not self._check_co_dependencies(x_raw, rtol=rtol):
            return False
        if not self._check_tcp_flags(x_raw):
            return False
        if attack_label is not None and not self._check_semantic(x_raw, attack_label):
            return False
        return True

    def _check_bounds(self, x: np.ndarray) -> bool:
        for name, (lo, hi) in self.bounds.items():
            if name in self._feat_idx:
                val = float(x[self._feat_idx[name]])
                if val < lo or val > hi:
                    return False
        return True

    def _check_co_dependencies(self, x: np.ndarray, rtol: float = 0.01) -> bool:
        for rule in self.co_dep_rules:
            if rule.derived not in self._feat_idx:
                continue
            residual = rule.residual(x, self._feat_idx)
            expected = rule.compute([float(x[self._feat_idx[s]]) for s in rule.sources])
            denom = max(abs(expected), _EPS)
            if residual / denom > rtol:
                return False
        return True

    def _check_tcp_flags(self, x: np.ndarray) -> bool:
        for col in self.tcp_flag_cols:
            flags = int(round(float(x[self._feat_idx[col]]))) & TCP_MAX
            if not is_valid_tcp_flags(flags):
                return False
        return True

    def _check_semantic(self, x: np.ndarray, attack_label: int) -> bool:
        for sc in self.semantic_constraints:
            if sc.attack_label == attack_label and sc.feature in self._feat_idx:
                if not sc.satisfied(x, self._feat_idx):
                    return False
        return True

    # ── CSR ───────────────────────────────────────────────────────────────

    def csr(
        self,
        x_batch: np.ndarray,
        attack_labels: np.ndarray | None = None,
        rtol: float = 0.01,
    ) -> float:
        """Compute Constraint Satisfaction Rate over a batch.

        Args:
            x_batch: Shape ``(batch, n_features)``. Raw-scale values.
            attack_labels: Shape ``(batch,)``. Optional per-sample attack
                labels for semantic preservation checks.
            rtol: Relative tolerance forwarded to ``check()``.

        Returns:
            CSR in [0.0, 1.0]. A value of 1.0 means every sample satisfies
            all constraints — the minimum bar for adversarial evaluation.
        """
        if len(x_batch) == 0:
            return 1.0
        satisfied = 0
        for i, x in enumerate(x_batch):
            label = int(attack_labels[i]) if attack_labels is not None else None
            if self.check(x, attack_label=label, rtol=rtol):
                satisfied += 1
        return satisfied / len(x_batch)

    # ── Degree Anomaly (Edge Injection) ───────────────────────────────────

    def check_degree_anomaly(
        self,
        original_degrees: np.ndarray,
        new_degrees: np.ndarray,
        sigma_multiplier: float = 3.0,
        train_mean: float | None = None,
        train_std: float | None = None,
    ) -> bool:
        """Check that injected edges do not create anomalous node degrees.

        Used exclusively by ``EdgeInjectionAttack``. Node degrees after
        injection must remain within ``sigma_multiplier`` standard deviations
        of the **training** degree distribution.

        Args:
            original_degrees: Degree array of all nodes before injection
                (shape: ``(num_nodes,)``).
            new_degrees: Degree array of all nodes after injection.
            sigma_multiplier: Number of standard deviations allowed above
                the training mean (default: 3.0 per spec).
            train_mean: Pre-computed mean of the training degree distribution.
                If ``None``, falls back to ``mean(original_degrees)`` — only
                valid when called on a clean (unattacked) graph.  Callers
                should compute and cache these stats from the training split
                once and pass them here to prevent threshold drift.
            train_std: Pre-computed std of the training degree distribution.
                Same semantics as ``train_mean``.

        Returns:
            True if no node's new degree exceeds the threshold.
        """
        mean = train_mean if train_mean is not None else float(np.mean(original_degrees))
        std = train_std if train_std is not None else float(np.std(original_degrees))
        if std == 0.0:
            std = 1.0  # avoid threshold == mean when all degrees are identical
        threshold = mean + sigma_multiplier * std
        return bool(np.all(new_degrees <= threshold))
