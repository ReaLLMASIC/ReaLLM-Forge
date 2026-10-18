"""Optional Transformers 4.44.2 Trainer bridge for native Muon + AdamW.

The production search uses the chunked-loss, exact-token CLI trainer. This
adapter is for standard HF Trainer integration; it does not override its loss
or make its token/VRAM accounting equivalent to the CLI experiment.
"""
import copy
import torch
from transformers import Trainer
from experiment.optimizers import optimizer_for


class MuonAdamW(torch.optim.Optimizer):
    """Expose the audited two-optimizer bundle as one scheduler-compatible optimizer."""

    def __init__(self, model, recipe):
        if recipe["weight_decay"] != 0.0:
            raise ValueError("This search fixes Muon weight_decay=0.0")
        self.bundle = optimizer_for(model, "muon", recipe, str(next(model.parameters()).device))
        # Reuse the live dictionaries: HF schedulers must update the same LR
        # values that the inner optimizers consume.
        super().__init__(self.bundle.param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.bundle.step()
        return loss

    def zero_grad(self, set_to_none=True):
        self.bundle.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.bundle.state_dict()

    def load_state_dict(self, state_dict):
        self.bundle.load_state_dict(state_dict)
        # Native optimizer.load_state_dict replaces its group dictionaries.
        self.param_groups = self.bundle.param_groups


class MuonTrainer(Trainer):
    """Build the optimizer after model_init/device placement, including HF HPO.

    recipe may be a dict or a callable(model.config) so each model_init trial
    can persist its own Muon/auxiliary-AdamW learning rates in the HF config.
    Do not pass a prebuilt optimizer when using model_init.
    """
    def __init__(self, *args, muon_recipe, **kwargs):
        self.muon_recipe = muon_recipe
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is None:
            recipe = self.muon_recipe(self.model.config) if callable(self.muon_recipe) else self.muon_recipe
            self.optimizer = MuonAdamW(self.model, copy.deepcopy(recipe))
        return self.optimizer
