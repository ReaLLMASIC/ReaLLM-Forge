# train_variations/optimizer_variants.py
from __future__ import annotations

import math
import torch
import itertools
from torch.optim import (ASGD, LBFGS, SGD, Adagrad, Adam, Adamax, AdamW, NAdam,
                         RAdam, RMSprop, SparseAdam)
from torch.optim.optimizer import Optimizer

try:
    from lion_pytorch import Lion
except ImportError:
    Lion = None

try:
    from apollo_torch import APOLLOAdamW          # pip install apollo-torch
except ImportError as e:                           # graceful fallback
    APOLLOAdamW = None

# Other PyPI optimisers ---------------------------------------------------
try:  # AdaBelief
    from adabelief_pytorch import AdaBelief
except ImportError:
    AdaBelief = None

try:  # Adan
    from adan_pytorch import Adan
except ImportError:
    Adan = None

# --------------------------------------------------------------------
#  Optional:  third-party optimisers (torch-optimizer)
#             imported lazily so the code runs even if the package
#             is absent and the user doesn’t choose those options.
# --------------------------------------------------------------------
try:
    import torch_optimizer as topt
except ModuleNotFoundError:  # keep rest of repo usable
    topt = None


def _needs_topt():
    if topt is None:
        raise ImportError(
            "This optimiser relies on the `torch-optimizer` package.\n"
            "➡  pip install torch-optimizer"
        )

# -------------------------------------------------------------------------
# Lookahead wrapper – works with *any* existing inner optimiser  (2025)
# Paper: M. Zhang et al., “Lookahead Optimizer: k steps forward, 1 step back”
# -------------------------------------------------------------------------
class Lookahead(Optimizer):
    r"""
    Generic Lookahead that wraps *any* PyTorch optimiser.

    Args
    ----
    optimizer : the inner/fast optimiser instance
    k         : number of inner steps before a slow update
    alpha     : interpolation factor  (0 < α ≤ 1).  α=0.5 ⇒ slow <- 0.5·slow + 0.5·fast
    """

    def __init__(self, optimizer: Optimizer, k: int = 6, alpha: float = 0.5):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if k < 1:
            raise ValueError("k must be ≥ 1")
        self.optimizer = optimizer
        self.k      = k
        self.alpha  = alpha
        self._step  = 0

        # Keep slow weights **outside** the inner optimiser’s state
        self._slow_buffers = [
            p.data.clone().detach() for p in
            itertools.chain.from_iterable(g["params"] for g in self.optimizer.param_groups)
        ]

    # --- mandatory PyTorch boilerplate -----------------------------------
    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self, set_to_none: bool = False):
        return self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        base = self.optimizer.state_dict()
        base.update({
            "lookahead_k":      self.k,
            "lookahead_alpha":  self.alpha,
            "lookahead_step":   self._step,
            "lookahead_slow":   [t.cpu() for t in self._slow_buffers],
        })
        return base

    def load_state_dict(self, state_dict):
        # 1) pull our own keys
        self.k     = int(state_dict.pop("lookahead_k",      self.k))
        self.alpha = float(state_dict.pop("lookahead_alpha", self.alpha))
        self._step = int(state_dict.pop("lookahead_step",   0))

        # checkpoints created **without** Lookahead won’t have slow buffers
        slow = state_dict.pop("lookahead_slow", None)
        if slow is None:
            slow = [p.data.clone().detach() for p in
                    itertools.chain.from_iterable(
                        g["params"] for g in self.optimizer.param_groups)]

        self._slow_buffers = [
            t.clone().to(p.device)
            for t, p in zip(
                slow,
                itertools.chain.from_iterable(g["params"]
                                              for g in self.optimizer.param_groups)
            )
        ]

        # 2) delegate the remainder to the inner optimiser
        self.optimizer.load_state_dict(state_dict)

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step += 1

        # 2) interpolate slow weights every *k* steps
        if self._step % self.k:
            return loss

        buf_idx = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                slow = self._slow_buffers[buf_idx]

                # ── keep slow buffer on same device as the live param ─────
                if slow.device != p.data.device:
                    slow = slow.to(p.data.device, non_blocking=True)
                    self._slow_buffers[buf_idx] = slow        # update ref

                slow.add_(p.data - slow, alpha=self.alpha)   # slow ← slow + α·Δ
                p.data.copy_(slow)                           # fast ← slow
                buf_idx += 1

        return loss

# -------------------------------------------------------------
def _lookahead(param_groups, args):
    """
    Wrap any existing optimiser in Lookahead.

    Extra CLI flags expected:
      --lookahead_inner_opt  adamw   (name of an optimiser already in optimiser_dictionary)
      --lookahead_k          6
      --lookahead_alpha      0.5
    """
    inner_name = getattr(args, "lookahead_inner_opt", "adamw")
    if inner_name == "lookahead":
        raise ValueError("Lookahead cannot wrap itself!")
    if inner_name not in optimizer_dictionary:
        raise ValueError(f"Unknown inner optimiser: {inner_name}")

    # Build the *inner* optimiser first, re-using param_groups + generic args
    inner_opt = optimizer_dictionary[inner_name](param_groups, args)

    # YAML gives us strings → cast defensively
    k      = int(getattr(args, "lookahead_k", 6))
    alpha  = float(getattr(args, "lookahead_alpha", 0.5))
    return Lookahead(inner_opt, k=k, alpha=alpha)



## Variance Adaptive LR
class VarianceAdaptiveLR(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-8, weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError("Learning rate must be positive")
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta value")

        defaults = dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta = group["beta"]
            eps = group["eps"]
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients not supported.")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["grad_avg_sq"] = torch.zeros_like(p.data)

                grad_avg_sq = state["grad_avg_sq"]
                state["step"] += 1

                grad_sq = grad * grad
                grad_avg_sq.mul_(beta).add_(grad_sq, alpha=1 - beta)

                # compute adaptive learning rate scaling
                adaptive_scale = lr / (grad_avg_sq.sqrt() + eps)

                if wd != 0:
                    p.data.mul_(1.0 - lr * wd)

                p.data.addcmul_(grad, adaptive_scale, value=-1.0)

        return loss

def _var_adaptive_lr(param_groups, args):
    return VarianceAdaptiveLR(
        param_groups,
        lr=args.learning_rate,
        beta=args.varlr_beta,
        eps=args.opt_eps,
        weight_decay=args.opt_weight_decay,
    )

# -------------------------------------------------------------------------
#  Lamb-DiffGrad  — layer-wise trust-ratio  ×  element-wise diff-friction
# -------------------------------------------------------------------------
class LambDiffGrad(Optimizer):
    r"""
    Implementation notes
    --------------------
    • Updates are Adam-style first/second moments  m, v
    • diffGrad friction  ξ = 1 / (1 + exp(-|g_t - g_{t-1}|))   (AbsSig)
      — element-wise:  big change ⇒ ξ≈1  (free step) ;
                       small change ⇒ ξ→0.5 (high friction)
    • Layer trust-ratio   τ = clamp(‖w‖ / (‖Δ‖ + 1e-12),  0,  clamp_value)

        Δ = ξ * m̂ / (√v̂ + eps)

      The final step is   w ← w  − lr * τ * Δ
    • AdamW-style decoupled weight decay (before trust-ratio)
    """
    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        clamp_value: float = 10.0,
        debias: bool = True,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if clamp_value <= 0:
            raise ValueError("clamp_value must be positive")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            clamp_value=clamp_value,
            debias=debias,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            clamp_value = group["clamp_value"]
            debias = group["debias"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                # State init -------------------------------------------------
                if not state:
                    state["step"] = 0
                    # Adam moments (float32 for numerical stability)
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    # diffGrad needs previous gradient
                    state["prev_grad"] = torch.zeros_like(p)

                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]
                prev_g: torch.Tensor = state["prev_grad"]

                state["step"] += 1
                t = state["step"]

                # Cast grads to float32 for the moment math
                g_fp32 = g.float()
                # diffGrad friction coefficient  ξ  -------------------------
                diff = (prev_g - g_fp32).abs()
                xi = 1.0 / (1.0 + torch.exp(-diff))          # range (0.5, 1)

                # Adam moments ---------------------------------------------
                exp_avg.mul_(beta1).addcmul_(g_fp32, xi, value=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_fp32, g_fp32, value=1 - beta2)

                if debias:
                    bias_c1 = 1 - beta1**t
                    bias_c2 = 1 - beta2**t
                    m_hat = exp_avg / bias_c1
                    v_hat = exp_avg_sq / bias_c2
                else:
                    m_hat, v_hat = exp_avg, exp_avg_sq

                update = m_hat / (v_hat.sqrt().add(eps))      # Adam step
                update.mul_(xi)                               # ← diffGrad term

                # Decoupled weight-decay *before* trust ratio
                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)

                # Layer-wise trust ratio -----------------------------------
                w_norm = p.data.norm(p=2)
                u_norm = update.norm(p=2)
                trust_ratio = (
                    w_norm / (u_norm + 1e-12)
                    if (w_norm > 0 and u_norm > 0)
                    else 1.0
                )
                trust_ratio = trust_ratio.clamp(max=clamp_value)

                p.data.add_(update, alpha=-lr * trust_ratio)

                # store current grad for next diffGrad step
                state["prev_grad"].copy_(g_fp32)

        return loss


# --- factory -------------------------------------------------------------
def _lambdiff(param_groups, args):
    """
    Instantiate the hybrid LambDiffGrad optimiser.

    Re-uses generic CLI flags:
        --learning_rate
        --opt_betas
        --opt_eps
        --opt_weight_decay
        --lamb_clamp         (max trust-ratio, default 10)
        --lamb_debias        (store bias-corr moments)
    """
    betas = tuple(args.opt_betas) if hasattr(args, "opt_betas") else (args.beta1, args.beta2)
    return LambDiffGrad(
        param_groups,
        lr=args.learning_rate,
        betas=betas,
        eps=args.opt_eps,
        weight_decay=args.opt_weight_decay,
        clamp_value=args.lamb_clamp,
        debias=args.lamb_debias,
    )

# ###############  AdEMAMix  ###################
#
# Paper ― “The AdEMAMix Optimizer: Better, Faster, Older” (Pagliardini et al., 2024)
# Key idea: keep *two* exponential moving averages of the gradient:
#   m1 – a fast EMA (β1 ≈ 0.9)           – retains responsiveness
#   m2 – a *very* slow EMA (β3 ≈ 0.9999) – stores decades-old grads
# Update uses a weighted mix  ( m1  +  α·m2 ) / √ν
# where ν is the usual 2-nd moment (AdamW-style).  α is warmed-up
# together with β3 to avoid early explosions.

class AdEMAMix(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999, 0.9999),        # (β1, β2, β3_final)
        alpha: float = 5.0,                # final α
        T_alpha_beta3: int = 0,            # warm-up steps (0 → off)
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        if any(not 0.0 <= b < 1.0 for b in betas):
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(
            lr=lr,
            betas=betas,
            alpha=alpha,
            T_alpha_beta3=T_alpha_beta3,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            beta1, beta2, beta3_final = group["betas"]
            alpha_final = group["alpha"]
            T = group["T_alpha_beta3"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                # --- state init ---
                if len(state) == 0:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(p)
                    state["m2"] = torch.zeros_like(p)
                    state["nu"] = torch.zeros_like(p)

                m1, m2, nu = state["m1"], state["m2"], state["nu"]
                state["step"] += 1
                t = state["step"]

                # --- linear warm-ups (disabled if T==0) ---
                if T > 0:
                    warm = min(1.0, t / T)
                    beta3 = beta1 + warm * (beta3_final - beta1)
                    alpha = warm * alpha_final
                else:
                    beta3, alpha = beta3_final, alpha_final

                # --- decoupled weight-decay (AdamW style) ---
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                # --- moment updates ---
                m1.mul_(beta1).add_(g, alpha=1 - beta1)          # fast EMA
                m2.mul_(beta3).add_(g, alpha=1 - beta3)          # slow EMA
                nu.mul_(beta2).addcmul_(g, g, value=1 - beta2)   # 2-nd moment

                # bias correction for m1 / nu (none for m2)
                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t

                denom = (nu.sqrt() / math.sqrt(bc2)).add_(eps)
                step_dir = (m1 / bc1 + alpha * m2) / denom
                p.add_(step_dir, alpha=-lr)
        return loss

# ---- factory helper ---------------------------------------
def _ademamix(param_groups, args):
    return AdEMAMix(
        param_groups,
        lr=args.learning_rate,
        betas=(args.ademamix_beta1, args.ademamix_beta2, args.ademamix_beta3),
        alpha=args.ademamix_alpha,
        T_alpha_beta3=args.ademamix_warmup,
        eps=args.opt_eps,
        weight_decay=args.weight_decay,
    )

################################################################################
# Sophia-G  (Diagonal Hessian + clipped Newton step)
# Paper: “Sophia: A Scalable Stochastic Second-order Optimizer for  
#         Language Modeling”  – Ma et al., 2023-24 (v4)
# URL :  https://arxiv.org/abs/2305.14342
################################################################################

class SophiaG(Optimizer):
    r"""
    **Sophia-G** keeps (1) a momentum buffer *m*  and (2) a *running diagonal
    Hessian* estimate *h*.  Every `update_freq` steps it refreshes *h*
    with the squared gradient; between refreshes it reuses the stale value,
    making the extra cost negligible in wall-clock terms.

       m_t   = β₁ m_{t-1} + (1-β₁) g_t
       h_t   = β₂ h_{t-1} + (1-β₂) g_t²          (only on refresh steps)
       Δ_t   = clip( m_t / (h_t + ε) ,  -ρ , ρ )
       θ_{t+1} = θ_t − η Δ_t      (  AdamW-style decoupled weight-decay first )

    Args
    ----
    params        : iterable of Tensors
    lr            : learning rate          (η)
    betas         : (β₁, β₂)               momentum & Hessian EMA factors
    rho           : clipping threshold     (ρ in the paper, default 0.04)
    update_freq   : refresh period for h_t (k  in the paper, default 10)
    eps           : numerical epsilon
    weight_decay  : decoupled weight decay
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.96, 0.99),
        rho: float = 0.04,
        update_freq: int = 10,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate")
        if update_freq < 1:
            raise ValueError("update_freq must be ≥ 1")
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            update_freq=update_freq,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            beta1, beta2 = group["betas"]
            rho, k = group["rho"], group["update_freq"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                state = self.state[p]

                # ── state init ────────────────────────────────────────────
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    state["h"] = torch.zeros_like(p, dtype=torch.float32)

                m, h = state["m"], state["h"]
                state["step"] += 1
                t = state["step"]

                # ── momentum ──────────────────────────────────────────────
                m.mul_(beta1).add_(g, alpha=1 - beta1)

                # ── Hessian diag update every k steps ────────────────────
                if t % k == 0:
                    h.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # ── decoupled weight-decay (AdamW style) ─────────────────
                if wd != 0.0:
                    p.data.mul_(1 - lr * wd)

                # ── pre-conditioned & clipped update ─────────────────────
                denom = h.add(eps)            # element-wise
                update = m / denom
                update.clamp_(min=-rho, max=rho)
                p.data.add_(update, alpha=-lr)
        return loss


################################################################################
# SOAP  –  “Shampoo plus Adam in the Eigen-basis”  (Mohan et al., 2024-25)
# Very memory-heavy but ~-40 % tokens-to-target on GPT-J / OPT-1.3 B.
#
# Implementation note
# -------------------
# •  We reuse torch-optimizer’s Kronecker-factored **Shampoo** to get the
#    pre-conditioned gradient `g̃`.  Inside SOAP we *also* keep an Adam-style
#    first-moment buffer on that pre-conditioned gradient – exactly as in the
#    paper (“momentum in the eigen-basis”).
# •  If `torch_optimizer.Shampoo` is unavailable we raise an ImportError with
#    a helpful hint (keeps repo importable even on minimal installs).
################################################################################

try:
    from torch_optimizer import Shampoo as _ToptShampoo
except (ModuleNotFoundError, ImportError):
    _ToptShampoo = None


class SOAP(Optimizer):
    r"""
    SOAP = **S**hampoo-**O**rthogonal **A**dam-grafted **P**rojection

    Stats per parameter
    -------------------
    •  _ToptShampoo keeps the Kronecker factors and their running inverses;
    •  we keep *one* extra momentum buffer **m** in the *Shampoo basis*.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),   # (β₁  for m,  β₂ for Shampoo)
        eps: float = 1e-12,
        weight_decay: float = 0.0,
        momentum: float = 0.9,        # reused for Shampoo
        update_freq: int = 1,         # Shampoo pre-condition freq
        graft_lr: float = 1.0,        # LR multiplier after grafting
    ):
        if _ToptShampoo is None:
            raise ImportError(
                "`torch-optimizer` not found.  Install with:\n"
                "    pip install torch-optimizer[shampoo]\n"
                "or choose a different optimiser."
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            update_freq=update_freq,
            graft_lr=graft_lr,
        )
        super().__init__(params, defaults)

        # A *real* Shampoo instance to handle Kronecker math & inverse roots
        self._shampoo = _ToptShampoo(
            params,
            lr=1.0,                       # handled by SOAP itself
            momentum=momentum,
            epsilon=eps,
            weight_decay=0.0,             # decay handled in outer loop
            update_freq=update_freq,
        )

        # we must NOT let torch-optimizer step the params – override later
        self._shampoo.step = lambda *a, **kw: None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()

        # 1) Let Shampoo compute **pre-conditioned grads** (stored on .grad)
        self._shampoo.precondition_grads()   # no param update here

        # 2) Adam-style first-moment + weight decay + param update
        for group in self.param_groups:
            lr, eps = group["lr"], group["eps"]
            beta1, _ = group["betas"]
            wd, η = group["weight_decay"], group["graft_lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.grad, dtype=torch.float32)
                m = state["m"]

                # Shampoo has already given us *g̃*  (pre-conditioned grad)
                g_tilde = p.grad

                # Momentum in the Shampoo eigen-basis
                m.mul_(beta1).add_(g_tilde, alpha=1 - beta1)

                # Decoupled weight decay
                if wd != 0.0:
                    p.data.mul_(1 - lr * wd)

                # Final update  (optionally graft lr)
                p.data.add_(m, alpha=-lr * η)

        return loss

# ──────────────────────────────────────────────────────────────────────────────
#  Factory helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sophiag(param_groups, args):
    return SophiaG(
        param_groups,
        lr=args.learning_rate,
        betas=(args.sophiag_beta1, args.sophiag_beta2),
        rho=args.sophiag_rho,
        update_freq=args.sophiag_update_freq,
        eps=args.opt_eps,
        weight_decay=args.opt_weight_decay,
    )


def _soap(param_groups, args):
    return SOAP(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.opt_eps,
        weight_decay=args.opt_weight_decay,
        momentum=args.shampoo_momentum,
        update_freq=args.shampoo_update_freq,
        graft_lr=args.soap_graft_lr,
    )

###############################################################################
# AdamS (Momentum-as-Normalizer) – Huishuai Zhang et al. 2024
# Paper: “Momentum Itself Can Be a Normalizer for LLM Pre-/Post-training”
#   • Denominator:  √( β₂ m² + (1-β₂) g² )
#   • One state tensor fewer than Adam/AdamW  → ½ memory
#   • Drop-in replacement for AdamW hyper-params
###############################################################################

class AdamS(Optimizer):
    r"""
    Implements **AdamS** from Zhang et al. (2024) –– a memory-lean variant of
    Adam that *removes* the second-moment buffer.  It matches SGD-momentum in
    memory footprint while retaining Adam-like adaptivity.

    https://arxiv.org/abs/2505.16363

    Args
    ----
    params         : iterable of parameters
    lr             : learning rate  (default 1e-3)
    betas          : (β1, β2)  momentum & denominator EMA factors
    eps            : numerical ε  (default 1e-8)
    weight_decay   : decoupled weight decay  (AdamW style)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate")
        if not 0.0 <= eps:
            raise ValueError("Invalid eps")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("AdamS does not support sparse gradients")

                state = self.state[p]
                # State initialisation ---------------------------------------------------
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)

                exp_avg: torch.Tensor = state["exp_avg"]
                state["step"] += 1

                # Weight decay (decoupled – same spot as AdamW)
                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)

                # Momentum --------------------------------------------------------------
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)

                # Denominator (no v buffer!)
                denom = (beta2 * exp_avg.pow(2) + (1 - beta2) * g.pow(2)).sqrt().add_(eps)

                step_size = lr
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

# -----------------------------------------------------------------------------
#  Factory helper – matches create_optimizer() convention
# -----------------------------------------------------------------------------

def _adams(param_groups, args):
    return AdamS(
        param_groups,
        lr=args.learning_rate,
        betas=(args.adams_beta1, args.adams_beta2),
        eps=args.adamw_eps,
        weight_decay=args.adamw_weight_decay,
    )


class OrthoAdam(Optimizer):
    """
    Memory-efficient implementation of the 'OrthoAdam' optimiser
    proposed in Kaul et al. 2024 (arXiv:2410.17174).
    For each parameter tensor we store **only**
      • a random permutation (int32) and
      • a ±1 sign vector (float32/float16),
    together representing an orthogonal matrix Q = diag(sign) · P.
    This keeps O(N) memory instead of O(N²).

    Args
    ----
    params : iterable of Tensors
    lr     : learning rate (η)
    betas  : (β1, β2)
    eps    : numerical eps
    weight_decay : AdamW-style decoupled weight decay
    permute_threshold : if tensor.numel() > threshold we skip rotation
                        (i.e. use identity Q) to avoid storing big buffers
    """

    # def __init__(self, params, *, lr=1e-3, betas=(0.9, 0.999),
    #              eps=1e-8, weight_decay=0.0, permute_threshold=1_000_000):
    def __init__(
        self,
        params,
        *,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        permute_threshold=1_000_000,
        tiny_threshold=128,
    ):
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            permute_threshold=permute_threshold,
        )
        super().__init__(params, defaults)

        # Build permutation & sign for each parameter once
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                n = p.numel()
                if n < tiny_threshold:  # tiny (bias/LN γ,β) → identity
                    state["perm"] = None
                    state["sign"] = None
                elif n > permute_threshold:  # huge → identity (memory)
                    state["perm"] = None
                    state["sign"] = None
                else:
                    device = p.device
                    state["perm"] = torch.randperm(n, device=device, dtype=torch.int64)
                    state["sign"] = (
                        torch.randint(0, 2, (n,), device=device, dtype=p.dtype) * 2.0
                        - 1.0
                    )

                # # Adam buffers in optimiser basis
                # state["step"] = 0
                # state["exp_avg"] = torch.zeros(n, device=p.device, dtype=p.dtype)
                # state["exp_avg_sq"] = torch.zeros(n, device=p.device, dtype=p.dtype)
                # Adam buffers (keep **always** in fp32 to avoid fp16 under-flow)
                state["step"] = 0
                state["exp_avg"] = torch.zeros(n, device=p.device, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros(
                    n, device=p.device, dtype=torch.float32
                )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("Sparse parameters not supported.")

                # Flatten to 1-D for permutation maths
                g = grad.view(-1)
                state = self.state[p]
                perm, sign = state["perm"], state["sign"]

                # # Weight decay in *parameter* basis
                # if wd != 0.0:
                #     g = g.add(p.data.view(-1), alpha=wd)

                # ---- rotate gradient to optimiser basis
                if perm is not None:  # Q = diag(sign)·P
                    # make sure buffers are on same device as grad
                    if perm.device != g.device:
                        perm = state["perm"] = perm.to(g.device, non_blocking=True)
                        sign = state["sign"] = sign.to(g.device, non_blocking=True)
                    g_bar = g[perm] * sign
                else:  # identity
                    g_bar = g

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # -------- keep moments on the same device as the grad
                if exp_avg.device != g.device:
                    exp_avg = state["exp_avg"] = exp_avg.to(g.device, non_blocking=True)
                    exp_avg_sq = state["exp_avg_sq"] = exp_avg_sq.to(
                        g.device, non_blocking=True
                    )

                state["step"] += 1
                step = state["step"]

                # Adam moment updates in optimiser basis
                exp_avg.mul_(beta1).add_(g_bar, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_bar, g_bar, value=1 - beta2)

                # Bias correction
                bias_c1 = 1 - beta1**step
                bias_c2 = 1 - beta2**step
                denom = (exp_avg_sq / bias_c2).sqrt().add_(eps)
                step_bar = (exp_avg / bias_c1) / denom

                # ---- rotate update back to parameter basis
                if perm is not None:
                    inv_perm = torch.empty_like(perm)
                    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)
                    step_vec = (step_bar * sign)[inv_perm]
                else:
                    step_vec = step_bar

                # Parameter update (in-place)
                # AdamW-style **decoupled** weight-decay _after_ inverse rotation
                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(step_vec.view_as(p.data), alpha=-lr)

        return loss


# Momentum / SGD family
def _sgd(param_groups, args):
    return SGD(
        param_groups,
        lr=args.learning_rate,
        momentum=args.sgd_momentum,
        nesterov=args.sgd_nesterov,
        weight_decay=args.weight_decay,
    )


# Adam family
def _adam(param_groups, args):
    return Adam(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
    )


def _adamw(param_groups, args):
    return AdamW(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adamw_eps,
        weight_decay=args.adamw_weight_decay,
    )


class ActRegularizedAdamW(AdamW):
    """AdamW with activation-aware weight decay.

    Instead of relying on gradients of the activations, this variant scales the
    weight-decay term by a statistic of the forward activations (e.g. overall
    standard deviation).  The statistic must be provided externally via
    :func:`set_activation_stat` prior to each :func:`step` call.
    """

    def __init__(self, params, *, activation_decay=0.0, stat_type="stdev", **kwargs):
        super().__init__(params, **kwargs)
        self.activation_decay = activation_decay
        self.stat_type = stat_type
        self._activation_stat: float = 0.0

    def set_activation_stat(self, stat: float) -> None:
        """Supply the activation statistic for regularisation."""
        self._activation_stat = float(stat)

    @torch.no_grad()
    def step(self, closure=None):
        if self.activation_decay != 0.0 and self._activation_stat != 0.0:
            coeff = self.activation_decay * self._activation_stat
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    p.grad = p.grad.add(p, alpha=coeff)

        loss = super().step(closure)
        self._activation_stat = 0.0
        return loss


def _adamw_act_reg(param_groups, args):
    return ActRegularizedAdamW(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adamw_eps,
        weight_decay=args.adamw_weight_decay,
        activation_decay=getattr(args, "activation_decay", 0.0),
        stat_type=getattr(args, "activation_stat", "stdev"),
    )

def _radam(param_groups, args):
    return RAdam(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
    )


def _adamax(param_groups, args):
    return Adamax(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
    )


def _nadam(param_groups, args):
    return NAdam(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.nadam_eps,
        weight_decay=args.weight_decay,
    )


# RMS / Ada family
def _rmsprop(param_groups, args):
    return RMSprop(
        param_groups,
        lr=args.learning_rate,
        alpha=args.rmsprop_alpha,
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
    )


def _rprop(param_groups, args):
    """
    Plain PyTorch Rprop.  We expose the multiplicative factors (etas) and
    re‑use them for step‑size bounds for simplicity.
    """
    return torch.optim.Rprop(
        param_groups,
        lr=args.learning_rate,
        etas=(args.rprop_eta_min, args.rprop_eta_max),
        step_sizes=(args.rprop_eta_min, args.rprop_eta_max),
    )


def _adagrad(param_groups, args):
    return Adagrad(
        param_groups,
        lr=args.learning_rate,
        lr_decay=args.adagrad_lr_decay,
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
        foreach=False,
    )


def _sparseadam(param_groups, args):
    return SparseAdam(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adamw_eps,
    )


# Second-order / others
def _asgd(param_groups, args):
    return ASGD(
        param_groups,
        lr=args.learning_rate,
        lambd=args.asgd_lambda,
        alpha=args.asgd_alpha,
        t0=args.asgd_t0,
        weight_decay=args.weight_decay,
    )


def _lbfgs(param_groups, args):
    return LBFGS(
        param_groups,
        lr=args.learning_rate,
        max_iter=args.lbfgs_max_iter,
        max_eval=args.lbfgs_max_eval,
        tolerance_grad=args.lbfgs_tol_grad,
        tolerance_change=args.lbfgs_tol_change,
        history_size=args.lbfgs_history,
        line_search_fn=args.lbfgs_line_search,
    )


# ---- Ortho optimisers ------------------------------------------------
def _orthoadam(param_groups, args):
    """Memory-cheap OrthoAdam (permute + sign)."""
    flat_params = [p for g in param_groups for p in g["params"]]
    return OrthoAdam(
        flat_params,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
        permute_threshold=getattr(args, "ortho_perm_threshold", 1_000_000),
    )

# Community contributed Optimizers

def _lion(param_groups, args):
    if Lion is None:
        raise ImportError("pip install lion-pytorch to use --optimizer lion")
    return Lion(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )


#    from adabelief_pytorch import AdaBelief
# AdaBelief ---------------------------------------------------------------
def _adabelief(param_groups, args):
    if AdaBelief is None:
        raise ImportError("pip install adabelief-pytorch  to use --optimizer adabelief")
    return AdaBelief(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adabelief_eps,
        weight_decay=args.weight_decay,
        rectify=False,  # keep as in the paper
    )


# Adan --------------------------------------------------------------------
def _adan(param_groups, args):
    if Adan is None:
        raise ImportError("pip install adan-pytorch  to use --optimizer adan")
    return Adan(
        param_groups,
        lr=args.learning_rate,
        betas=(
            args.beta1,
            args.beta2,
            args.adan_beta3 if hasattr(args, "adan_beta3") else 0.999,
        ),
        eps=args.adan_eps,
        weight_decay=args.adan_wd,
    )





# Yogi --------------------------------------------------------------------
def _yogi(param_groups, args):
    if Yogi is None:
        raise ImportError("pip install pytorch_optimizer  to use --optimizer yogi")
    return Yogi(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
    )




# Torch-Optimizer


# ---------- torch-optimizer wrappers --------------------
def _accsgd(param_groups, args):
    _needs_topt()
    return topt.AccSGD(
        param_groups,
        lr=args.learning_rate,
        kappa=1000,
        xi=10,
        small_const=0.7,
        weight_decay=args.opt_weight_decay,
    )


def _adabound(param_groups, args):
    _needs_topt()
    return topt.AdaBound(
        param_groups,
        lr=args.learning_rate,
        betas=tuple(args.opt_betas),
        final_lr=args.adabound_final_lr,
        gamma=args.adabound_gamma,
        eps=args.opt_eps,
        weight_decay=args.opt_weight_decay,
    )


def _adamod(param_groups, args):
    _needs_topt()
    return topt.AdaMod(
        param_groups,
        lr=args.learning_rate,
        betas=tuple(args.opt_betas),
        beta3=args.adamod_beta3,
        eps=args.opt_eps,
        weight_decay=args.opt_weight_decay,
    )


def _adamp(param_groups, args):
    _needs_topt()
    return topt.AdamP(
            param_groups,
            lr=args.learning_rate,
            betas=tuple(args.opt_betas),
            eps=args.opt_eps,
            weight_decay=args.opt_weight_decay
        )

def _adafactor(param_groups, args):
    _needs_topt()
    # When lr is None Adafactor uses the “relative step” schedule from
    # the paper (√t scaling + warm-up).  If the user provided
    # --learning_rate we forward it, otherwise stay in the default mode.
    lr_kw = {} if args.learning_rate is None else {"lr": args.learning_rate}
    return topt.Adafactor(
        param_groups,
        **lr_kw,
        eps2=(args.adafactor_eps_row, args.adafactor_eps_col),
        clip_threshold=args.adafactor_clip,
        decay_rate=args.adafactor_decay,
        beta1=(None if args.adafactor_beta1 < 0 else args.adafactor_beta1),
        weight_decay=args.opt_weight_decay,
        scale_parameter=args.adafactor_scale_param,
        relative_step=args.adafactor_relative_step,
        warmup_init=args.adafactor_warmup_init,
    )



def _aggmo(param_groups, args):
    _needs_topt()
    return topt.AggMo(
        param_groups, lr=args.learning_rate, betas=(0.0, 0.9, 0.99), weight_decay=args.opt_weight_decay
    )


def _diffgrad(param_groups, args):
    _needs_topt()
    return topt.DiffGrad(
        param_groups,
        lr=args.learning_rate,
        betas=tuple(args.opt_betas),
        eps=args.opt_eps,
        weight_decay=args.opt_weight_decay,
    )


def _lamb(param_groups, args):
    _needs_topt()
    return topt.Lamb(
        param_groups,
        lr=args.learning_rate,
        betas=tuple(args.opt_betas),
        eps=args.opt_eps,
        weight_decay=args.opt_weight_decay,
        clamp_value=args.lamb_clamp,
        adam=args.lamb_adam,
        debias=args.lamb_debias,
    )


def _novograd(param_groups, args):
    _needs_topt()
    return topt.NovoGrad(
        param_groups,
        lr=args.learning_rate,
        betas=tuple(args.opt_betas),
        eps=args.opt_eps,
        weight_decay=args.opt_weight_decay,
    )


def _pid(param_groups, args):
    _needs_topt()
    return topt.PID(
        param_groups,
        lr=args.learning_rate,
        momentum=args.pid_momentum,
        integral=args.pid_integral,
        derivative=args.pid_derivative,
        weight_decay=args.opt_weight_decay,
    )

def _qhadam(param_groups, args):
    _needs_topt()
    return topt.QHAdam(
            param_groups,
            lr=args.learning_rate,
            betas=tuple(args.opt_betas),
            nus=(1.0, 1.0),
            eps=args.opt_eps,
            weight_decay=args.opt_weight_decay
        )

def _qhm(param_groups, args):
    _needs_topt()
    return topt.QHM(
        param_groups,
        lr=args.learning_rate,
        momentum=args.sgd_momentum,  # reuse flag
        nu=0.7,
        weight_decay=args.opt_weight_decay,
    )


def _sgdp(param_groups, args):
    _needs_topt()
    return topt.SGDP(
        param_groups,
        lr=args.learning_rate,
        momentum=args.sgd_momentum,
        eps=args.opt_eps,
        weight_decay=args.opt_weight_decay,
    )


def _sgdw(param_groups, args):
    _needs_topt()
    return topt.SGDW(
        param_groups,
        lr=args.learning_rate,
        momentum=args.sgd_momentum,
        weight_decay=args.opt_weight_decay,
        nesterov=args.sgd_nesterov,
    )


def _shampoo(param_groups, args):
    _needs_topt()
    return topt.Shampoo(
        param_groups,
        lr=args.learning_rate,
        momentum=args.shampoo_momentum,
        weight_decay=args.opt_weight_decay,
        epsilon=args.shampoo_eps,
        update_freq=args.shampoo_update_freq,
    )


def _swats(param_groups, args):
    _needs_topt()
    return topt.SWATS(
        param_groups,
        lr=args.learning_rate,
        betas=tuple(args.opt_betas),
        eps=args.opt_eps,
        weight_decay=args.opt_weight_decay,
    )

def _yogi(param_groups, args):
    _needs_topt()
    return topt.Yogi(
            param_groups,
            lr=args.learning_rate,
            betas=tuple(args.opt_betas),
            eps=args.opt_eps,
            weight_decay=args.opt_weight_decay,
        )


## Apollo


def _apollo_adamw(param_groups, args):
    """
    Returns an APOLLOAdamW optimiser that shares the *same* generic
    hyper-parameters (lr / betas / eps / weight_decay) as the rest of your
    Adam-family variants, while adding Apollo’s low-rank options.

    Required extra CLI flags
    ------------------------
    --apollo_rank                int      rank of the low-rank projector
    --apollo_proj                str      ['random' | 'hadamard' | 'learned']
    --apollo_scale               int      scaling constant for projector
    --apollo_update_proj_gap     int      #steps between projector refresh
    --apollo_proj_type           str      ['std' | 'gaussian' | 'rademacher']
    """
    if APOLLOAdamW is None:
        raise ImportError(
            "Package `apollo-torch` is not installed. "
            "Run `pip install apollo-torch` or switch optimiser."
        )

    ############################################################
    # 1) Split the incoming *PyTorch* param groups into
    #    “low-rank” and “regular” groups so that users can still
    #    choose which tensors use the special update.
    ############################################################
    # A parameter is tagged “low-rank” if it has attribute `lowrank=True`
    # **or**  if user requested global low-rank via --apollo_apply_to_all.
    low_groups, reg_groups = [], []
    apply_to_all = getattr(args, "apollo_apply_to_all", False)

    for pg in param_groups:
        lr_params, reg_params = [], []
        for p in pg["params"]:
            if getattr(p, "lowrank", False) or apply_to_all:
                lr_params.append(p)
            else:
                reg_params.append(p)

        # preserve original group hyper-params
        shared = {k: v for k, v in pg.items() if k != "params"}

        if reg_params:
            reg_groups.append({"params": reg_params, **shared})
        if lr_params:
            low_groups.append(
                {
                    "params": lr_params,
                    "rank": args.apollo_rank,
                    "proj": args.apollo_proj,
                    "scale_type": "tensor",
                    "scale": args.apollo_scale,
                    "update_proj_gap": args.apollo_update_proj_gap,
                    "proj_type": args.apollo_proj_type,
                    **shared,
                }
            )

    # Stitch the list back together –– Apollo can accept mixed groups.
    pg = reg_groups + low_groups

    ############################################################
    # 2) Instantiate the optimiser
    ############################################################
    opt = APOLLOAdamW(
        pg,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
    )
    return opt

class AdaModDiffGrad(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999, 0.9999),  # (beta1, beta2, beta3)
        eps=1e-8,
        weight_decay=0.0,
    ):
        if lr <= 0.0:
            raise ValueError("Learning rate must be positive")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    state["eta_avg"] = torch.zeros_like(p.data)
                    state["prev_grad"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                eta_avg, prev_grad = state["eta_avg"], state["prev_grad"]

                state["step"] += 1

                # diffGrad friction coefficient ξ
                diff = (grad - prev_grad).abs()
                xi = 1.0 / (1.0 + torch.exp(-diff))

                # update moments
                exp_avg.mul_(beta1).addcmul_(xi, grad, value=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_c1 = 1 - beta1**state["step"]
                bias_c2 = 1 - beta2**state["step"]

                m_hat = exp_avg / bias_c1
                v_hat = exp_avg_sq / bias_c2

                eta = lr / (v_hat.sqrt().add(eps))

                # AdaMod bounding
                eta_avg.mul_(beta3).add_(eta, alpha=1 - beta3)
                eta_bound = torch.min(eta, eta_avg)

                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                p.data.addcmul_(m_hat, eta_bound, value=-1.0)

                prev_grad.copy_(grad)

        return loss

def _adamod_diffgrad(param_groups, args):
    return AdaModDiffGrad(
        param_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2, args.adamod_beta3),
        eps=args.opt_eps,
        weight_decay=args.opt_weight_decay,
    )


class Muon(Optimizer):
    """Simplified Muon optimizer with weight decay and update scaling."""

    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.0,
                 ns_iters=5, scale=0.2):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_iters=ns_iters, scale=scale)
        super().__init__(params, defaults)

    def _newton_schulz(self, M, iters):
        a, b, c = 3.4445, -4.7750, 2.0315
        eps = 1e-8
        X = M / (M.norm(p="fro") + eps)
        for _ in range(iters):
            XtX = X @ X.transpose(-1, -2)
            X = a * X + b * (XtX @ X) + c * (XtX @ XtX @ X)
        return X

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            iters = group["ns_iters"]
            scale = group["scale"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p.data)

                m = state["momentum"]
                m.mul_(mu).add_(grad)

                update = m
                if m.ndim >= 2 and m.size(0) > 0 and m.size(1) > 0:
                    O = self._newton_schulz(m, iters)
                    factor = max(m.size(0), m.size(1)) * scale
                    update = O * factor

                if wd != 0.0:
                    update = update + wd * p.data

                p.data.add_(update, alpha=-lr)

        return loss


def _muon(param_groups, args):
    return Muon(
        param_groups,
        lr=args.learning_rate,
        momentum=getattr(args, "muon_momentum", 0.95),
        weight_decay=args.weight_decay,
        ns_iters=getattr(args, "muon_ns_iters", 5),
        scale=getattr(args, "muon_scale", 0.2),
    )


optimizer_dictionary: dict[str, callable] = {
    # From pytorch
    "sgd": _sgd,
    "adam": _adam,
    "adamw": _adamw,
    "adamw_act_reg": _adamw_act_reg,
    "adamax": _adamax,
    "radam": _radam,
    "nadam": _nadam,
    "adagrad": _adagrad,
    "rmsprop": _rmsprop,
    "rprop": _rprop,
    "sparseadam": _sparseadam,
    "asgd": _asgd,
    "lbfgs": _lbfgs,
    # paper-driven implementations
    "orthoadam": _orthoadam,
    "adams": _adams,
    "ademamix": _ademamix,
    # hybrids
    "lambdiff": _lambdiff,
    "adamod_diffgrad": _adamod_diffgrad,
    # community contributed
    "lion": _lion,
    # from adabelief_pytorch
    "adabelief": _adabelief,
    # from adan pytorch
    "adan": _adan,
    # apollo_pytotrch
    "apollo_adamw": _apollo_adamw,
    # from torch-optimizer suite
    "adamp": _adamp,
    "adafactor": _adafactor,
    "accsgd": _accsgd,
    "adabound": _adabound,
    "adamod": _adamod,
    "aggmo": _aggmo,
    "diffgrad": _diffgrad,
    "lamb": _lamb,
    "novograd": _novograd,
    "pid": _pid,
    "qhm": _qhm,
    "sgdp": _sgdp,
    "sgdw": _sgdw,
    "shampoo": _shampoo,
    "swats": _swats,
    "qhadam": _qhadam,
    "yogi": _yogi,
    "sophiag": _sophiag,
    "soap": _soap,
    "muon": _muon,
    "var_adaptive_lr": _var_adaptive_lr,
    "lookahead": _lookahead,
}
