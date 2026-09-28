"""AIME's production statistics update for the drift experiments: responsibility-gated gains
tau_k = tau * min(1, rho_k / rho_ref) and, optionally, change-point retention from shs_rssm/forgetting.py."""
from __future__ import annotations

import torch

FACTORS = ("emission", "transition", "gate")
DYN_KEYS = ("N", "Srr", "Szr", "Szz", "Shh", "Szh", "Srh")
FACTOR_OF = dict({k: "emission" for k in DYN_KEYS}, start="emission", C="transition", pg_A="gate", pg_h="gate")
RETAINED = set(DYN_KEYS) | {"C", "pg_A", "pg_h"}


def shares(mass):
    mass = mass.to(torch.float64).clamp_min(0.0)
    return (mass * mass.shape[0] / mass.sum().clamp_min(1e-12)).clamp(0.0, 1.0)


class RetentionRule:
    def __init__(self, forgetting, tau, retain=True):
        self.fg, self.tau, self.retain = forgetting, float(tau), bool(retain)
        self.drift = {f: forgetting.DriftPosterior() for f in FACTORS} if retain else None
        self.usage, self.gain = {}, {}

    def gains(self, message):
        u_k, u_row = shares(message["N"]), shares(message["C"].sum(-1))
        self.usage = dict(emission=u_k, transition=u_row, gate=u_row)
        self.gain = {f: self.tau * u for f, u in self.usage.items()}
        return self.gain

    def evidence(self, core, total, message):
        fg, g = self.fg, self.gain
        weights = {f: fg.stream_weights(g[f], self.tau) for f in FACTORS}
        return dict(
            emission=fg.EmissionEvidence(core.dyn, {k: total[k] for k in DYN_KEYS},
                                         {k: message[k] for k in DYN_KEYS}, *weights["emission"]),
            transition=fg.TransitionEvidence(core.prior_dest, total["C"], message["C"], *weights["transition"]),
            gate=fg.GateEvidence(core.gate.sigma0_diag, core.gate.m0, total["pg_A"], total["pg_h"],
                                 message["pg_A"], message["pg_h"], *weights["gate"]))

    def factors(self, core, total, message, learn):
        self.gains(message)
        if not self.retain:
            return {}, {}
        r, info = {}, {}
        for name, ev in self.evidence(core, total, message).items():
            r[name], info[name] = self.fg.retention(ev, self.usage[name], self.drift[name], learn=learn)
        return r, info

    def propose(self, total, message, r):
        out = {}
        for key, v in message.items():
            shape = (-1,) + (1,) * (v.dim() - 1)
            g = self.gain[FACTOR_OF[key]].to(v.dtype).reshape(shape)
            keep = 1.0 - g
            if key in RETAINED and FACTOR_OF[key] in r:
                keep = keep * r[FACTOR_OF[key]].to(v.dtype).reshape(shape)
            out[key] = keep * total[key] + g * v
        return out


def summary(rule, r, info):
    """Rows (mean retention, largest drift probability, drift rate, mean exposure, failed regimes) per factor."""
    rows = []
    for f in FACTORS:
        u = float(rule.usage[f].mean())
        if f in r:
            rows.append([float(r[f].mean()), float(info[f]["P"].max()), float(info[f]["rate"]), u,
                         float(info[f]["failed"])])
        else:
            rows.append([1.0, 0.0, float("nan"), u, 0.0])
    return rows
