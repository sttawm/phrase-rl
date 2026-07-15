"""CRN phrase scoring with a frozen LeRobot pi0 policy.

pi0's flow-matching loss is stochastic in the noise eps and flow time tau that
LeRobot samples inside forward(). For phrase scoring we need common random
numbers: every phrase of a context is scored under the SAME K (eps, tau) draws,
so loss differences reflect conditioning only. LeRobot's PI0Policy.forward
accepts optional `noise` and `time` tensors (verified at runtime), so no
monkey-patching is required — we pass fixed draws explicitly.

tau is stratified over (tau_min, 1): K bins, one uniform sample per bin
(LeRobot's internal Beta sampling is a training choice, not required for
scoring; see EXPERIMENT.md "Reward"). Default tau_min=0.0 keeps all pre-Phase-2
results bit-identical; training passes tau_min=0.25 (0b finding).

Score of phrase y_i for context (o, state, a*):
    L_i = (1/K) sum_k mean over (horizon, action_dim) of
          || v_theta(A^tau_k; o, y_i, tau_k) - u*_k ||^2
Reported per (phrase, draw) so callers can do split-half reliability and
per-draw z-scoring downstream.
"""

import numpy as np
import torch


def import_pi0_policy():
    """INTACT-era lerobot (IrvingF7 fork @35f6e02) uses the old package layout."""
    try:
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy  # old layout
    except ModuleNotFoundError:
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy  # >= 0.4 layout
    return PI0Policy


def make_draws(k: int, horizon: int, action_dim: int, seed: int, tau_min: float = 0.0):
    """K fixed (eps, tau) draws, deterministic in seed. tau stratified over (tau_min, 1)."""
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(k, horizon, action_dim, generator=g)
    tau = (torch.arange(k, dtype=torch.float32) + torch.rand(k, generator=g)) / k
    # 0b finding: discriminability collapses for tau<0.25 (clean-action end) — Phase 2 passes tau_min=0.25
    tau = tau_min + (1.0 - tau_min) * tau  # identity when tau_min=0.0 (exact in fp32)
    # keep away from exact 0/1 endpoints (pi0 internals assume open interval)
    tau = tau.clamp(1e-3, 1 - 1e-3)
    return noise, tau


class Pi0PhraseScorer:
    """Scores phrase lists against one context with common random numbers."""

    def __init__(self, policy, k: int = 16, seed: int = 0, micro_batch: int = 64,
                 tau_min: float = 0.0):
        self.policy = policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)
        self.device = next(policy.parameters()).device
        cfg = policy.config
        self.image_keys = list(cfg.image_features)
        self.horizon = cfg.chunk_size
        self.action_dim = cfg.action_feature.shape[0]  # raw actions in the batch (7)
        # pi0 pads actions to max_action_dim internally; noise must match the padded shape
        padded_dim = getattr(cfg, "max_action_dim", self.action_dim)
        self.k = k
        self.seed = seed
        self.padded_dim = padded_dim
        self.micro_batch = micro_batch
        self.noise, self.tau = make_draws(k, self.horizon, padded_dim, seed, tau_min)

    def _per_sample_loss(self, batch, noise, time):
        """Forward with fixed draws; return per-sample loss (B,)."""
        loss, loss_dict = self.policy.forward(
            batch, noise=noise.to(self.device), time=time.to(self.device)
        )
        losses = loss_dict.get("losses_after_forward")
        if losses is None:
            raise RuntimeError(
                f"per-sample losses not in loss_dict (keys={list(loss_dict)}); "
                "lerobot version mismatch — adapt _per_sample_loss"
            )
        return losses.float().mean(dim=tuple(range(1, losses.ndim)))

    @torch.no_grad()
    def score(self, image_chw: np.ndarray, state: np.ndarray, action_chunk: np.ndarray,
              phrases: list[str]) -> np.ndarray:
        """Returns losses of shape (n_phrases, K). image_chw: float32 (3,H,W) in [0,1]."""
        P, K = len(phrases), self.k
        img = torch.from_numpy(image_chw)
        st = torch.from_numpy(state.astype(np.float32))
        act = torch.from_numpy(action_chunk.astype(np.float32).reshape(self.horizon, self.action_dim))

        # flat layout: sample b = (phrase p, draw k) with k fastest
        out = torch.empty(P, K)
        pairs = [(p, k) for p in range(P) for k in range(K)]
        for start in range(0, len(pairs), self.micro_batch):
            chunk = pairs[start : start + self.micro_batch]
            B = len(chunk)
            batch = {
                "observation.state": st.expand(B, -1).to(self.device),
                "action": act.expand(B, -1, -1).to(self.device),
                "task": [phrases[p] for p, _ in chunk],
            }
            for key in self.image_keys:
                batch[key] = img.expand(B, -1, -1, -1).to(self.device)
            noise = torch.stack([self.noise[k] for _, k in chunk])
            time = torch.stack([self.tau[k] for _, k in chunk])
            losses = self._per_sample_loss(batch, noise, time)
            for (p, k), l in zip(chunk, losses.cpu()):
                out[p, k] = l
        return out.numpy()

    @torch.no_grad()
    def score_verbose(self, image_chw: np.ndarray, state: np.ndarray, action_chunk: np.ndarray,
                      phrases: list[str]):
        """Like score(), but also returns the raw velocity vectors per draw.

        Returns (losses (P,K), v_pred (P,K,H,7), u_target (P,K,H,7)) — first 7
        (real) action dims only; padding dims dropped. v_theta is captured with a
        forward hook on the velocity head (model.action_out_proj); u = noise - a_norm
        is recomputed by mirroring forward()'s normalize/pad path and verified
        against losses_after_forward on every micro-batch (fail-loud on mismatch).
        """
        proj = getattr(self.policy.model, "action_out_proj", None)
        assert proj is not None, "pi0 layout changed: model.action_out_proj missing — adapt score_verbose"
        P, K, H = len(phrases), self.k, self.horizon
        img = torch.from_numpy(image_chw)
        st = torch.from_numpy(state.astype(np.float32))
        act = torch.from_numpy(action_chunk.astype(np.float32).reshape(H, self.action_dim))

        captured = []
        handle = proj.register_forward_hook(lambda m, i, o: captured.append(o.detach()))
        out_l = torch.empty(P, K)
        out_v = torch.empty(P, K, H, self.action_dim)
        out_u = torch.empty(P, K, H, self.action_dim)
        try:
            pairs = [(p, k) for p in range(P) for k in range(K)]
            for start in range(0, len(pairs), self.micro_batch):
                chunk = pairs[start : start + self.micro_batch]
                B = len(chunk)
                batch = {
                    "observation.state": st.expand(B, -1).to(self.device),
                    "action": act.expand(B, -1, -1).to(self.device),
                    "task": [phrases[p] for p, _ in chunk],
                }
                for key in self.image_keys:
                    batch[key] = img.expand(B, -1, -1, -1).to(self.device)
                noise = torch.stack([self.noise[k] for _, k in chunk]).to(self.device)
                time = torch.stack([self.tau[k] for _, k in chunk]).to(self.device)
                captured.clear()
                losses = self._per_sample_loss(batch, noise, time)
                assert len(captured) == 1, f"expected 1 action_out_proj call, got {len(captured)}"
                v = captured[0][:, -H:, :].float()  # (B, H, padded)
                # u = noise - normalized padded actions (mirror of forward's target path)
                norm = self.policy.normalize_targets(dict(batch))
                a_n = norm["action"]
                if a_n.shape[-1] < v.shape[-1]:  # pad to match
                    a_n = torch.nn.functional.pad(a_n, (0, v.shape[-1] - a_n.shape[-1]))
                u = noise - a_n  # (B, H, padded)
                recon = ((u - v) ** 2).mean(dim=(1, 2))
                if not torch.allclose(recon, losses.to(recon.device), rtol=2e-2, atol=1e-4):
                    raise RuntimeError(
                        f"velocity reconstruction mismatch (recon {recon[:3].tolist()} vs "
                        f"forward {losses[:3].tolist()}) — padded-dim/masking convention changed"
                    )
                for j, (p, k) in enumerate(chunk):
                    out_l[p, k] = losses[j].cpu()
                    out_v[p, k] = v[j, :, : self.action_dim].cpu()
                    out_u[p, k] = u[j, :, : self.action_dim].cpu()
        finally:
            handle.remove()
        return out_l.numpy(), out_v.numpy(), out_u.numpy()

    @torch.no_grad()
    def decode_verbose(self, image_chw: np.ndarray, state: np.ndarray, action_chunk: np.ndarray,
                       phrases: list[str], dim_std: np.ndarray, k_l2: int = 4):
        """Like score_l2(), but also returns raw decoded chunks (P, k_l2, H, 7) unnormalized."""
        policy, model = self.policy, self.policy.model
        P, H = len(phrases), self.horizon
        a_star = torch.from_numpy(
            action_chunk.astype(np.float32).reshape(H, self.action_dim)).to(self.device)
        std = torch.from_numpy(np.asarray(dim_std, dtype=np.float32)).to(self.device)
        img = torch.from_numpy(image_chw).to(self.device)
        st = torch.from_numpy(state.astype(np.float32)).to(self.device)
        batch = {"observation.state": st.expand(P, -1), "task": list(phrases)}
        for key in self.image_keys:
            batch[key] = img.expand(P, -1, -1, -1)
        norm = policy.normalize_inputs(batch)
        images, img_masks = policy.prepare_images(norm)
        state_p = policy.prepare_state(norm)
        lang_tokens, lang_masks = policy.prepare_language(norm)
        g = torch.Generator().manual_seed(self.seed)
        dnoise = torch.randn(k_l2, H, self.padded_dim, generator=g)
        out_dec = torch.empty(P, k_l2, H, self.action_dim)
        out_l2 = torch.empty(P, k_l2)
        out_gr = torch.empty(P, k_l2)
        for k in range(k_l2):
            noise_k = dnoise[k][None].expand(P, -1, -1).to(self.device)
            acts = model.sample_actions(images, img_masks, lang_tokens, lang_masks, state_p, noise=noise_k)
            acts = policy.unnormalize_outputs({"action": acts[:, :, : self.action_dim]})["action"]
            nrm = ((acts - a_star) / std)[:, :, :6]
            out_l2[:, k] = torch.sqrt((nrm ** 2).mean(dim=(1, 2))).cpu()
            out_gr[:, k] = (acts - a_star)[:, :, 6].abs().mean(dim=1).cpu()
            out_dec[:, k] = acts.cpu()
        return out_dec.numpy(), out_l2.numpy(), out_gr.numpy()

    @torch.no_grad()
    def score_l2(self, image_chw: np.ndarray, state: np.ndarray, action_chunk: np.ndarray,
                 phrases: list[str], dim_std: np.ndarray, k_l2: int = 4) -> np.ndarray:
        """CRN decoded-action reward: returns per-dim-normalized L2 of shape (n_phrases, k_l2).

        For each of k_l2 FIXED noise draws (shared across all phrases -> common random
        numbers), run pi0's full denoising (model.sample_actions) for every phrase and
        take the per-DoF-normalized L2 to the ground-truth chunk over the 6 continuous
        DoF (gripper excluded, matching pi0_decode.norm_l2). Lower is better, so the
        trainer's R = -mean_k loss and z-advantages work identically to the flow arm.
        """
        policy, model, cfg = self.policy, self.policy.model, self.policy.config
        P = len(phrases)
        a_star = torch.from_numpy(
            action_chunk.astype(np.float32).reshape(self.horizon, self.action_dim)
        ).to(self.device)  # (H, 7)
        std = torch.from_numpy(np.asarray(dim_std, dtype=np.float32)).to(self.device)  # (7,)

        img = torch.from_numpy(image_chw).to(self.device)
        st = torch.from_numpy(state.astype(np.float32)).to(self.device)
        batch = {"observation.state": st.expand(P, -1), "task": list(phrases)}
        for key in self.image_keys:
            batch[key] = img.expand(P, -1, -1, -1)
        norm = policy.normalize_inputs(batch)
        images, img_masks = policy.prepare_images(norm)
        state_p = policy.prepare_state(norm)
        lang_tokens, lang_masks = policy.prepare_language(norm)

        # decode noise: k_l2 draws of the padded shape, deterministic in seed, shared across phrases
        g = torch.Generator().manual_seed(self.seed)
        dnoise = torch.randn(k_l2, self.horizon, self.padded_dim, generator=g)
        out = torch.empty(P, k_l2, device=self.device)
        for k in range(k_l2):
            noise_k = dnoise[k][None].expand(P, -1, -1).to(self.device)  # (P, H, padded)
            acts = model.sample_actions(images, img_masks, lang_tokens, lang_masks, state_p, noise=noise_k)
            acts = policy.unnormalize_outputs({"action": acts[:, :, : self.action_dim]})["action"]  # (P, H, 7)
            nrm = ((acts - a_star) / std)[:, :, :6]  # continuous DoF only
            out[:, k] = torch.sqrt((nrm ** 2).mean(dim=(1, 2)))
        return out.cpu().numpy()
