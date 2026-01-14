import torch
import torch.nn as nn

class LSVBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lsv_index = 0    # Dataset index default to zero
        self.lsv_dataset_num = config.lsv_dataset_num
        self.lsv_embd_dim = config.n_embd
        self.lsv_scaling_factor = 1.0
        self.mode = 1
        self.mixture = []

    def update_lsv_scaling_factor(self, new_scaling_factor):
        self.lsv_scaling_factor = new_scaling_factor

    def get_lsv_scaling_factor(self):
        return self.lsv_scaling_factor

    def update_lsv_index(self, new_index):
        self.lsv_index = new_index

    def set_mixture(self, mixture_list):
        """ for variation to override """
        self.mixture = mixture_list
        pass

    def set_mode(self, mode):
        """ Modes, generally:
        1 = one hot
        2 = mixture mode (set mixture and forget)
        """
        self.mode = mode

    def forward(self, x):
        return x

class OneHotLSV(LSVBase):
    """
    Refactored to use nn.Embedding. This is the idiomatic PyTorch way to
    handle lookups. It's more efficient and automatically handles sparse
    gradients, so only the selected vector is updated during training.
    """
    def __init__(self, config):
        super().__init__(config)
        self.lsv_embedding = nn.Embedding(self.lsv_dataset_num, config.n_embd)
        torch.nn.init.normal_(self.lsv_embedding.weight, mean=0.00, std=0.02)
        self.mixture_weights = torch.zeros(self.lsv_dataset_num, device=self.device)

    def set_mixture(self, mixture_list):
        """ mixture of different vectors """
        self.mixture_weights = torch.tensor(mixture_list, device=self.device)
        print("mixture weights set to:", self.mixture_weights)

    def forward(self, x):
        if self.mode == 1:
            # Directly look up the embedding vector for the current index.
            lsv_index_tensor = torch.tensor(self.lsv_index, dtype=torch.long, device=self.device)
            selected_vector = self.lsv_embedding(lsv_index_tensor) * self.lsv_scaling_factor
        else: # Mixture mode
            # Perform a weighted sum of all embedding vectors
            selected_vector = torch.matmul(self.mixture_weights, self.lsv_embedding.weight)

        # Add the selected vector to the input tensor x
        x = x + selected_vector
        return x

class LinearCombinationLSV(LSVBase):
    """
    Refactored to use two nn.Embedding layers.
    1. `lsv_embedding`: Stores the base vectors (previously `lsv_matrix`).
    2. `linear_comb_embedding`: Stores the combination weights for each dataset
       (previously `linear_comb_matrix`).
    This approach is much cleaner and more efficient.
    """
    def __init__(self, config):
        super().__init__(config)
        # Embedding table for the base LSV vectors
        self.lsv_embedding = nn.Embedding(self.lsv_dataset_num, config.n_embd)
        torch.nn.init.normal_(self.lsv_embedding.weight, mean=0.00, std=0.02)

        # Embedding table for the linear combination weights
        self.linear_comb_embedding = nn.Embedding(self.lsv_dataset_num, self.lsv_dataset_num)
        torch.nn.init.normal_(self.linear_comb_embedding.weight, mean=0.00, std=0.02)

    def forward(self, x):
        # Look up the combination weight vector for the current dataset index
        lsv_index_tensor = torch.tensor(self.lsv_index, dtype=torch.long, device=self.device)
        selected_linear_comb_vector = self.linear_comb_embedding(lsv_index_tensor) * self.lsv_scaling_factor

        # Use the looked-up vector to create a weighted sum of all base lsv vectors
        combined_vector = torch.matmul(selected_linear_comb_vector, self.lsv_embedding.weight)

        # Add the combined vector to the input tensor x
        x = x + combined_vector
        return x

class OneHotMLPLSV_Gating(LSVBase):
    """OneHotMLPLSV with gating mechanism using sigmoid activation."""
    def __init__(self, config):
        super().__init__(config)
        mlp_width = 64
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, mlp_width),
                nn.ReLU(),
                nn.Linear(mlp_width, mlp_width),
                nn.ReLU(),
                nn.Linear(mlp_width, config.n_embd)
            ).to(self.device) for _ in range(self.lsv_dataset_num)
        ])
        # Router produces gating values between 0 and 1 for each MLP
        self.router = nn.Sequential(
            nn.Linear(config.n_embd, self.lsv_dataset_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, seq_length, n_embd = x.size()
        x_flat = x.view(-1, n_embd)
        gates = self.router(x_flat)
        combined_output = torch.zeros_like(x_flat)
        for i, mlp in enumerate(self.mlps):
            mlp_output = mlp(x_flat)
            gate = gates[:, i].unsqueeze(-1)
            combined_output += gate * mlp_output
        combined_output = combined_output.view(batch_size, seq_length, n_embd)
        x = x + combined_output
        return x

class MixtureOfLSV(LSVBase):
    """ A FIRE Inspired method for combining LSVs with a learned router. """
    def __init__(self, config):
        super().__init__(config)
        mlp_width = 64
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, mlp_width),
                nn.ReLU(),
                nn.Linear(mlp_width, mlp_width),
                nn.ReLU(),
                nn.Linear(mlp_width, config.n_embd)
            ).to(self.device) for _ in range(self.lsv_dataset_num)
        ])
        # Define the learned router, which will output a probability distribution over MLPs
        self.router = nn.Sequential(
            nn.Linear(config.n_embd, self.lsv_dataset_num),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        batch_size, seq_length, n_embd = x.size()
        x_flat = x.view(-1, n_embd)
        router_probs = self.router(x_flat)
        combined_output = torch.zeros_like(x_flat, device=self.device)
        for i, mlp in enumerate(self.mlps):
            mlp_output = mlp(x_flat)
            prob = router_probs[:, i].unsqueeze(-1)
            combined_output += prob * mlp_output
        combined_output = combined_output.view(batch_size, seq_length, n_embd)
        x = x + combined_output
        return x

class OneHotMLPLSV(LSVBase):
    """
    A FIRE Inspired method for combining LSVs. The manual freezing logic
    has been removed, as PyTorch's autograd will not compute gradients for
    modules that are not used in the forward pass.
    """
    def __init__(self, config):
        super().__init__(config)
        mlp_width = 64
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, mlp_width),
                nn.ReLU(),
                nn.Linear(mlp_width, mlp_width),
                nn.ReLU(),
                nn.Linear(mlp_width, config.n_embd)
            ).to(self.device) for _ in range(self.lsv_dataset_num)
        ])

    def forward(self, x):
        if self.mode == 1:
            # Only the selected MLP is used, so only its parameters will receive gradients.
            selected_mlp = self.mlps[self.lsv_index]
            mlp_output = selected_mlp(x)
        else: # Mixture Mode
            mlp_output = 0
            for i in range(len(self.mlps)):
                mlp_output += self.mlps[i](x) * self.mixture[i]
        x = x + mlp_output
        return x

class OneHotMLPLSV_TopK(LSVBase):
    """OneHotMLPLSV with Top-K selection of MLPs."""
    def __init__(self, config, k=2):
        super().__init__(config)
        self.k = k
        mlp_width = 64
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, mlp_width),
                nn.ReLU(),
                nn.Linear(mlp_width, config.n_embd)
            ).to(self.device) for _ in range(self.lsv_dataset_num)
        ])
        self.router = nn.Linear(config.n_embd, self.lsv_dataset_num)

    def forward(self, x):
        batch_size, seq_length, n_embd = x.size()
        x_flat = x.view(-1, n_embd)
        router_logits = self.router(x_flat)
        _, topk_indices = torch.topk(router_logits, self.k, dim=-1)
        mask = torch.zeros_like(router_logits)
        mask.scatter_(1, topk_indices, 1.0)
        # Use straight-through estimator
        gates = (mask - router_logits.detach()) + router_logits
        combined_output = torch.zeros_like(x_flat)
        for i, mlp in enumerate(self.mlps):
            gate = gates[:, i].unsqueeze(-1)
            # This check avoids unnecessary computation
            if gate.abs().sum() > 0:
                mlp_output = mlp(x_flat)
                combined_output += gate * mlp_output
        combined_output = combined_output.view(batch_size, seq_length, n_embd)
        x = x + combined_output
        return x

class OneHotMLPLSV_MoE(LSVBase):
    """OneHotMLPLSV using Mixture of Experts with learned routing."""
    def __init__(self, config, router_temperature=1.0):
        super().__init__(config)
        self.router_temperature = router_temperature
        mlp_width = 64
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, mlp_width),
                nn.ReLU(),
                nn.Linear(mlp_width, config.n_embd)
            ).to(self.device) for _ in range(self.lsv_dataset_num)
        ])
        self.router = nn.Linear(config.n_embd, self.lsv_dataset_num)

    def forward(self, x):
        batch_size, seq_length, n_embd = x.size()
        x_flat = x.view(-1, n_embd)
        router_logits = self.router(x_flat) / self.router_temperature
        router_probs = nn.functional.softmax(router_logits, dim=-1)
        combined_output = torch.zeros_like(x_flat)
        for i, mlp in enumerate(self.mlps):
            mlp_output = mlp(x_flat)
            prob = router_probs[:, i].unsqueeze(-1)
            combined_output += prob * mlp_output
        combined_output = combined_output.view(batch_size, seq_length, n_embd)
        x = x + combined_output
        return x

class OneHotMLPLSV_Attention(LSVBase):
    """OneHotMLPLSV with attention-based routing."""
    def __init__(self, config):
        super().__init__(config)
        mlp_width = 64
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, mlp_width),
                nn.ReLU(),
                nn.Linear(mlp_width, config.n_embd)
            ).to(self.device) for _ in range(self.lsv_dataset_num)
        ])
        # Learnable queries
        self.queries = nn.Parameter(torch.randn(self.lsv_dataset_num, config.n_embd))

    def forward(self, x):
        batch_size, seq_length, n_embd = x.size()
        x_flat = x.view(-1, n_embd)
        # Compute attention scores
        attention_scores = torch.matmul(x_flat, self.queries.t())
        router_probs = nn.functional.softmax(attention_scores, dim=-1)
        combined_output = torch.zeros_like(x_flat)
        for i, mlp in enumerate(self.mlps):
            mlp_output = mlp(x_flat)
            prob = router_probs[:, i].unsqueeze(-1)
            combined_output += prob * mlp_output
        combined_output = combined_output.view(batch_size, seq_length, n_embd)
        x = x + combined_output
        return x

lsv_dictionary = {
    "one_hot": OneHotLSV,
    "linear_comb": LinearCombinationLSV,
    "one_hot_mlp": OneHotMLPLSV,
    "ohmg": OneHotMLPLSV_Gating,
    "ohmt": OneHotMLPLSV_TopK,
    "ohmm": OneHotMLPLSV_MoE,
    "ohma": OneHotMLPLSV_Attention,
    "mol": MixtureOfLSV,
}
