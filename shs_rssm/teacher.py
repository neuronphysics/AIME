"""EMA teacher encoder for structure moves under a training encoder.

Implements the second of Michael Hughes' three suggestions (email, Aug 2026):

    "you could possibly track statistics for two versions of the model: a solid
     reference and a evolving version (conceptually related to teacher/student
     networks)"

Why this and not the other two
------------------------------
Hughes' framing of the problem: memoized inference "requires an underlying
reference distribution that isn't changing at all. In a strict sense, this is the
only way to guarantee that the memoized statistics that represent the sum of
batch-specific quantities actually represent the whole dataset, and can therefore
be used to justify birth/merge/delete decisions."

His third suggestion, down-weighting stale summaries, is deliberately NOT
implemented. A decayed sum of statistics computed under different encoders is not
the whole-dataset sufficient statistic of anything, so the accept/reject test stops
bounding a fixed objective -- the pseudo-bound regime his 2015 paper exists to
escape. Discarding (purge_stale) keeps the guarantee; decaying does not.

His first suggestion, a trust region on the encoder, is a constraint on the world
model rather than on this module, and directly opposes the reconstruction
objective. It remains open.

What this does
--------------
Maintains theta_teacher <- (1 - tau) * theta_teacher + tau * theta_student.
The switching model consumes TEACHER latents only, so the reference distribution
the memoized statistics describe moves on the EMA timescale (~1/tau updates)
rather than the student's. Structure moves then compare summaries computed under
encoders that differ by an amount you control, instead of by an amount set by the
optimizer.

The teacher is not differentiated through: the regime model provides no gradient
to the encoder. Representation shape is driven by reconstruction/reward/continue
alone and the regime model is a passive observer. That is a real modelling
concession, taken deliberately -- it removes the coupling that made the corpus
non-stationary in the first place. A differentiable variant (student latents for
the ELBO term, teacher latents for move statistics) is a later option and is NOT
what this class does.

Contract with MoveBuffer
------------------------
`version` increments only when the teacher moves beyond `drift_tol` on a fixed
probe batch, not on every student step. Batches are tagged with that version, so
`MoveBuffer.purge_stale(teacher.version)` drops only summaries whose reference
distribution actually shifted. `safe_for_moves()` is the gate: it refuses a sweep
when accumulated drift since the last move exceeds tolerance, which is cheaper
than accepting a move scored against a moving reference.

Usage
-----
    teacher = EMATeacher(encoder, tau=0.005, probe=probe_batch)
    ...
    teacher.update(encoder)                      # every student step
    z = teacher.encode(obs)                      # feed the switching model
    buf.add(z, ..., repr_version=teacher.version)
    if teacher.safe_for_moves():
        sweep_moves(head, buffer=buf, lap=lap)
        teacher.mark_move()
"""
import copy

import torch


class EMATeacher:
    """Slow-moving copy of an encoder, with drift accounting for move scheduling."""

    def __init__(self, student, tau: float = 0.005, probe=None,
                 drift_tol: float = 0.05, device=None):
        """
        tau       EMA rate. Effective horizon ~1/tau updates; 0.005 -> ~200.
                  Move interval should be >> that horizon.
        probe     fixed batch used to measure drift in OUTPUT space. Parameter-space
                  distance is not a usable proxy: encoders can move a lot in weights
                  and little in outputs, and the statistics only care about outputs.
        drift_tol mean per-dimension L2 drift, in units of the probe's own std, that
                  is tolerated before the version bumps / moves are refused.
        """
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

    # ------------------------------------------------------------------ core
    @torch.no_grad()
    def _encode_raw(self, x):
        out = self.teacher(x)
        return (out[0] if isinstance(out, (tuple, list)) else out).detach()

    @torch.no_grad()
    def update(self, student):
        """One EMA step. Buffers are copied outright, not blended.

        Running BatchNorm statistics are not parameters and blending them is not
        meaningful; a stale buffer with fresh weights is a silent source of drift
        that the probe would attribute to the EMA.
        """
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

    # ------------------------------------------------------- drift accounting
    @torch.no_grad()
    def _refresh_drift(self):
        cur = self._encode_raw(self.probe)
        d = float((cur - self._ref).pow(2).mean().sqrt()) / self._probe_scale
        self._drift_since_move = d
        if d > self.drift_tol:
            self.version += 1          # reference distribution has materially moved
            self._ref = cur
            self._drift_since_move = 0.0

    def drift(self) -> float:
        """Output-space drift since the last version bump, in probe-std units."""
        return self._drift_since_move

    def safe_for_moves(self) -> bool:
        """Whether accumulated drift is small enough to score moves against.

        A skipped sweep costs nothing; a merge accepted against a shifting
        reference costs the run.
        """
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
