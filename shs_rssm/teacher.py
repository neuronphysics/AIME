import copy

import torch


class EMATeacher:
    """Slow-moving copy of an encoder, with drift accounting for move scheduling."""

    def __init__(self, student, tau: float = 0.005, probe=None,
                 drift_tol: float = 0.05, device=None):
        self.teacher = copy.deepcopy(student).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.teacher.to(device)
        self.tau = float(tau)
        self.probe = probe
        self.drift_tol = float(drift_tol)
        self.version = 0
        self._n_updates = 0
        self._ref = None if probe is None else self._encode_raw(probe)
        self._probe_scale = None if self._ref is None else \
            float(self._ref.std().clamp_min(1e-6))
        self._drift_since_move = 0.0

    @torch.no_grad()
    def _encode_raw(self, x):
        out = self.teacher(x)
        return (out[0] if isinstance(out, (tuple, list)) else out).detach()

    @torch.no_grad()
    def update(self, student):
        for pt, ps in zip(self.teacher.parameters(), student.parameters()):
            pt.mul_(1.0 - self.tau).add_(ps.detach(), alpha=self.tau)
        for bt, bs in zip(self.teacher.buffers(), student.buffers()):
            bt.copy_(bs)
        self.teacher.eval()
        self._n_updates += 1
        if self.probe is not None:
            self._refresh_drift()
        return self

    @torch.no_grad()
    def encode(self, x):
        """Teacher latents. Detached: no gradient path to the student."""
        return self._encode_raw(x)

    @torch.no_grad()
    def _refresh_drift(self):
        cur = self._encode_raw(self.probe)
        d = float((cur - self._ref).pow(2).mean().sqrt()) / self._probe_scale
        self._drift_since_move = d
        if d > self.drift_tol:
            self.version += 1
            self._ref = cur
            self._drift_since_move = 0.0

    def drift(self) -> float:
        """Output-space drift since the last version bump, in probe-std units."""
        return self._drift_since_move

    def safe_for_moves(self) -> bool:
        if self.probe is None:
            return True
        return self._drift_since_move <= self.drift_tol

    def mark_move(self):
        """Call after a completed sweep to reset the drift window."""
        if self.probe is not None:
            self._ref = self._encode_raw(self.probe)
        self._drift_since_move = 0.0
        return self

    @property
    def horizon(self) -> float:
        """Approximate number of student updates the teacher averages over."""
        return 1.0 / max(self.tau, 1e-12)

    def state_dict(self):
        return {"teacher": self.teacher.state_dict(), "tau": self.tau,
                "version": self.version, "n_updates": self._n_updates}

    def load_state_dict(self, sd):
        self.teacher.load_state_dict(sd["teacher"])
        self.tau = sd.get("tau", self.tau)
        self.version = sd.get("version", 0)
        self._n_updates = sd.get("n_updates", 0)
        if self.probe is not None:
            self._ref = self._encode_raw(self.probe)
            self._probe_scale = float(self._ref.std().clamp_min(1e-6))
        return self
