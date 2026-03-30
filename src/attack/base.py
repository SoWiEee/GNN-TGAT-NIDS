"""
src/attack/base.py
Abstract base class for all CAAG adversarial example generation methods.

All concrete attack implementations (CPGDAttack, EdgeInjectionAttack,
GANAttack) must inherit BaseAttack. This allows eval/comparison.py to drive
every attack through a unified interface via Hydra instantiate.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.base import BaseNIDSModel


class BaseAttack(ABC):
    """Unified interface for all CAAG adversarial example generators.

    Responsibilities of each concrete subclass:
    1. Generate adversarial examples that maximise attack success rate.
    2. Call ``ConstraintSet.project()`` after every gradient/generation step
       (inverse-transform → project → re-transform).
    3. Return **only** examples with CSR = 1.0 for final evaluation.

    The ``memory_reset_policy`` and ``target_split`` fields must be set via
    the Hydra config (``configs/attack/<method>.yaml``).
    """

    # ── Abstract interface ────────────────────────────────────────────────

    @abstractmethod
    def generate(self, model: "BaseNIDSModel", data, **kwargs):
        """Generate adversarial examples against ``model``.

        Args:
            model: Target NIDS model. White-box attacks have full access to
                weights and gradients; black-box attacks treat it as an oracle.
            data: Input graph data — PyG ``Data`` (static) or ``TemporalData``
                (temporal). The returned object must have the same type.
            **kwargs: Attack-specific runtime overrides (e.g. ``epsilon``).

        Returns:
            Perturbed graph data object. Contains only adversarial examples
            whose ``constraint_check()`` returns ``True`` (CSR = 1.0).
        """
        ...

    @abstractmethod
    def constraint_check(self, x_adv, attack_label: int | None = None) -> bool:
        """Return True only if **all** constraints in the constraint set are met.

        This is the per-sample gate that enforces CSR = 1.0: only samples
        passing this check are included in the final evaluation.

        Args:
            x_adv: Raw-scale (inverse-transformed) feature vector or batch.
            attack_label: Optional integer attack class used for semantic
                preservation checks (e.g. DDoS must retain high packet rate).
                Pass ``None`` to skip semantic enforcement.

        Returns:
            ``True`` if every constraint (protocol validity, co-dependency,
            feature bounds, semantic preservation) is satisfied.
        """
        ...

    # ── Provided helpers ──────────────────────────────────────────────────

    def batch_csr(self, x_batch, attack_labels=None) -> float:
        """Compute Constraint Satisfaction Rate over a batch.

        Delegates per-sample checking to ``constraint_check()``.

        Args:
            x_batch: Iterable of raw-scale feature vectors.
            attack_labels: Optional iterable of integer attack class labels,
                forwarded to ``constraint_check`` for semantic preservation.

        Returns:
            CSR in [0.0, 1.0]. 1.0 means every sample satisfies all constraints.
        """
        items = list(x_batch)
        if not items:
            return 1.0
        labels: list[int | None] = (
            list(attack_labels) if attack_labels is not None else [None] * len(items)
        )
        satisfied = sum(
            1 for x, lbl in zip(items, labels)
            if self.constraint_check(x, lbl)
        )
        return satisfied / len(items)
