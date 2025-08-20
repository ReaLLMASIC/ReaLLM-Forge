# 1.8 Mixture of Experts (MoE)

This section provides a comprehensive guide to all arguments related to the Mixture of Experts (MoE) architecture. MoE is a technique for increasing the number of parameters in a model without proportionally increasing the computational cost.

## Core MoE Configuration

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_moe` | bool | False | A global switch to enable the Mixture of Experts architecture. If `True`, some of the FFN layers will be replaced with MoE layers. |
| `--moe_layer_freq` | int | 2 | The frequency at which to replace FFNs with MoE layers. For example, a value of 2 means that every second FFN layer will be an MoE layer. |
| `--n_experts` | int | 8 | The total number of experts in each MoE layer. |
| `--moe_top_k` | int | 2 | The number of experts to route each token to. A higher value for `top_k` means that more experts are used for each token, which can improve performance but also increases computational cost. |
| `--moe_router_scheme` | str | "softmax" | The routing scheme to use for the MoE layer. Defaults to softmax. |