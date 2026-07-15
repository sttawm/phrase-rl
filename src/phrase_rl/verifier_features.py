"""Shared featurization for the learned verifier (reward model).

Input rows follow build_verifier_data.py's schema: flow_loss (K,), flow_v /
flow_u (K*H*7,), decoded (Kd*H*7,), norm_l2 / grip_err (Kd,), a_star (H*7,).
tau is slot-constant by CRN construction, so no tau input — per-slot weights
encode it (see WRITEUP "verifier" notes).

Modes: flow (raw v, u, v-u + per-slot loss + residual spread),
       l2   (raw decoded, a*, diff + norm_l2/grip + decode spread),
       both (concat).
"""

import numpy as np

K_FLOW, K_DEC, H, D = 8, 4, 4, 7


def _stack(col):
    return np.stack([np.asarray(x, dtype=np.float32) for x in col])


def featurize(df, mode: str, include_diff: bool = True) -> np.ndarray:
    parts = []
    if mode in ("flow", "both"):
        v = _stack(df.flow_v)                      # (N, K*H*D)
        u = _stack(df.flow_u)
        r = v - u                                  # residual, the raw thing MSE averages
        rs = r.reshape(-1, K_FLOW, H * D)
        parts += [v, u] + ([r] if include_diff else []) + [
                  _stack(df.flow_loss),            # per-slot scalar loss (K,)
                  rs.std(axis=1)]                  # residual spread across slots (H*D,)
    if mode in ("l2", "both"):
        dec = _stack(df.decoded)                   # (N, Kd*H*D)
        a = _stack(df.a_star)                      # (N, H*D)
        dd = dec.reshape(-1, K_DEC, H * D)
        diff = (dd - a[:, None, :]).reshape(len(a), -1)
        parts += [dec, a] + ([diff] if include_diff else []) + [
                  _stack(df.norm_l2), _stack(df.grip_err),
                  dd.std(axis=1)]                  # decode spread = policy uncertainty (H*D,)
    if mode == "deploy":
        # DEMO-FREE: no a*-derived features — usable at deployment on tasks with no
        # Bridge demo (test tiers). v is meaningful alone because eps is CRN-fixed.
        v = _stack(df.flow_v)
        vs = v.reshape(-1, K_FLOW, H * D)
        dec = _stack(df.decoded)
        dd = dec.reshape(-1, K_DEC, H * D)
        parts = [v, vs.std(axis=1), dec, dd.std(axis=1)]
    if not parts:
        raise ValueError(f"unknown mode {mode!r}")
    return np.concatenate(parts, axis=1)


def feature_dim(mode: str) -> int:
    n = 0
    if mode in ("flow", "both"):
        n += 3 * K_FLOW * H * D + K_FLOW + H * D
    if mode in ("l2", "both"):
        n += K_DEC * H * D + H * D + K_DEC * H * D + 2 * K_DEC + H * D  # dec, a*, diff, l2+grip, spread
    return n
