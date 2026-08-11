"""Locked training reward: calibrated 5-seed ensemble of verifier_both_nodiff.

Loads the manifest written at lock time (results/checkpoints/
verifier_reward_ensemble.json), reconstructs each member (MLP + feature
standardizer + logit shift + temperature), and scores verifier-schema feature
rows: reward logit = mean over members of (logit + shift) / T. The trainer
minimizes "loss", so the server emits loss = -logit (lower = better phrase).

CRITICAL: feature extraction must match training slots — k_flow=8, k_decode=4,
seed=0, tau_min=0.0 (the manifest records these; the server pins them and
ignores the job spec's k/tau_min in verifier mode).
"""

import json

import numpy as np
import torch

from phrase_rl.train_verifier import MLP
from phrase_rl.verifier_features import featurize


class VerifierEnsemble:
    def __init__(self, manifest_path: str):
        man = json.load(open(manifest_path))
        self.k_flow = int(man.get("k_flow", 8))
        self.k_decode = int(man.get("k_decode", 4))
        self.seed = int(man.get("seed", 0))
        self.tau_min = float(man.get("tau_min", 0.0))
        self.members = []
        for m in man["members"]:
            ck = torch.load(m["ckpt"], weights_only=False, map_location="cpu")
            net = MLP(ck["d_in"])
            net.load_state_dict(ck["state_dict"])
            net.eval()
            self.members.append({
                "net": net, "mu": ck["mu"], "sd": ck["sd"],
                "mode": ck["mode"], "include_diff": ck.get("include_diff", True),
                "shift": float(m["logit_shift"]), "T": float(m["temperature"]),
            })

    @torch.no_grad()
    def member_logits(self, feats_df) -> np.ndarray:
        """(P, n_members) calibrated logits for verifier-schema feature rows."""
        outs = []
        for m in self.members:
            X = (featurize(feats_df, m["mode"], m["include_diff"]) - m["mu"]) / m["sd"]
            z = m["net"](torch.from_numpy(X.astype(np.float32))).numpy()
            outs.append((z + m["shift"]) / m["T"])
        return np.stack(outs, axis=1)

    @torch.no_grad()
    def member_embeddings(self, feats_df) -> np.ndarray:
        """(P, 64) penultimate-layer activations, mean over members.

        The verifier-v2 residual head consumes these instead of the scalar
        logit: one number per frame is a lossy bottleneck on what the members
        know (the reason the current reward needs grip bolted on externally)."""
        outs = []
        for m in self.members:
            X = (featurize(feats_df, m["mode"], m["include_diff"]) - m["mu"]) / m["sd"]
            h = m["net"].net[:-1](torch.from_numpy(X.astype(np.float32))).numpy()
            outs.append(h)
        return np.mean(np.stack(outs, axis=0), axis=0)
