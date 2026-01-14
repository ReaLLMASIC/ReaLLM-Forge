# initializations/jl_transform_ckpt.py
import argparse
import os
import shutil
import math
import torch
import datetime
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply a Johnson-Lindenstrauss transform to all weights in a checkpoint"
    )
    parser.add_argument(
        "ckpt_dir",
        type=str,
        help="Directory containing ckpt.pt and meta.pkl from a previous training run",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to write the transformed checkpoint (defaults to <ckpt_dir>_jl)",
    )
    parser.add_argument(
        "--out_embd",
        type=int,
        required=True,
        help="Embedding dimension of the output checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for the JL transform",
    )
    parser.add_argument(
        "--jl_type",
        choices=["sign", "gaussian", "sparse", "srht", "qr"],
        default="gaussian",
        help="Type of JL transform: 'sign', 'gaussian', 'sparse' (Achlioptas), or 'srht'",
    )
    parser.add_argument(
        "--gaussian_mean",
        type=float,
        default=0.0,
        help="Mean of the normal distribution for the gaussian JL transform",
    )
    parser.add_argument(
        "--gaussian_std",
        type=float,
        default=1.0,
        help="Standard deviation of the normal distribution for the gaussian JL transform",
    )
    parser.add_argument(
        "--cproj_vertical",
        action="store_true",
        help="Project c_proj weights along the out_features dimension instead of the in_features dimension",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview shapes & stats but DO NOT write checkpoint",
    )
    parser.add_argument(
        "--proj_out",
        type=str,
        default=None,
        help="File to which the projection matrix will be saved (torch.save)",
    )
    parser.add_argument(
        "--proj_in",
        type=str,
        default=None,
        help="Load a pre-computed projection matrix instead of sampling one",
    )
    return parser.parse_args()

console = Console()

def sign_matrix(out_dim: int, in_dim: int, generator: torch.Generator, device) -> torch.Tensor:
    """Create a sign-based JL projection matrix of shape (out_dim, in_dim)."""
    mat = torch.randint(0, 2, (out_dim, in_dim), generator=generator, device=device, dtype=torch.float32)
    mat = mat * 2 - 1
    mat /= math.sqrt(out_dim)
    return mat


def jl_project_tensor(
    tensor: torch.Tensor, proj: torch.Tensor, vertical_only: bool = False
) -> torch.Tensor:
    """Project ``tensor`` using ``proj``.

    If ``vertical_only`` is True the projection is applied along the first
    dimension only (useful for ``c_proj`` weights when projecting the
    out_features).  Otherwise the last dimension and, when appropriate, the first
    dimension are projected so that tensors mapping the residual dimension to
    itself are resized correctly.
    """
    in_dim = proj.shape[1]

    if tensor.ndim == 0:
        return tensor

    if vertical_only:
        if tensor.ndim > 1 and tensor.shape[0] == in_dim:
            return proj @ tensor
        if tensor.ndim == 1 and tensor.shape[0] == in_dim:
            return (proj @ tensor.unsqueeze(-1)).squeeze(-1)
        return tensor

    if tensor.ndim >= 1 and tensor.shape[-1] == in_dim:
        tensor = tensor @ proj.t()

    if tensor.ndim > 1 and tensor.shape[0] == in_dim:
        tensor = proj @ tensor
    elif tensor.ndim == 1 and tensor.shape[0] == in_dim:
        tensor = (proj @ tensor.unsqueeze(-1)).squeeze(-1)

    return tensor


def main():
    args = parse_args()

    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    meta_path = os.path.join(args.ckpt_dir, "meta.pkl")

    console.rule("[bold cyan]Loading checkpoint")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", checkpoint)

    # optimizer and scheduler states depend on parameter shapes.
    # They will be invalid after changing the embedding dimension, so drop them.
    checkpoint.pop("optimizer", None)
    checkpoint.pop("scheduler", None)

    g = torch.Generator()
    g.manual_seed(args.seed)

    # Determine the embedding dimension to transform
    old_embd = checkpoint.get("model_args", {}).get("n_embd")
    if old_embd is None and "config" in checkpoint:
        old_embd = checkpoint["config"].get("n_embd")
    if old_embd is None:
        raise ValueError("Could not determine n_embd from checkpoint")

    config = checkpoint.get("config", {})
    n_head = config.get("n_head")
    old_qk_dim = config.get("n_qk_head_dim")
    old_v_dim = config.get("n_v_head_dim")
    if n_head is not None:
        if old_qk_dim is None:
            old_qk_dim = old_embd // n_head
        if old_v_dim is None:
            old_v_dim = old_embd // n_head


    if args.proj_in:
        console.print(f"[green]Loading projection matrix from[/] {args.proj_in}")
        proj = torch.load(args.proj_in, map_location="cpu")
        if proj.shape != (args.out_embd, old_embd):
            raise ValueError(f"Loaded proj shape {proj.shape} != ({args.out_embd}, {old_embd})")
    elif args.jl_type == "gaussian":
        proj = torch.empty((args.out_embd, old_embd), device="cpu")
        proj.normal_(mean=args.gaussian_mean,
                 std=args.gaussian_std,
                 generator=g)
        proj /= math.sqrt(args.out_embd)
    elif args.jl_type == "sparse":
        rand = torch.rand(args.out_embd, old_embd, generator=g, device="cpu")
        proj = torch.zeros_like(rand)
        proj[rand < 1/6] = 1
        proj[(rand >= 1/6) & (rand < 2/6)] = -1
        proj *= math.sqrt(3.0 / args.out_embd)
    elif args.jl_type == "srht":
        if (old_embd & (old_embd - 1)) != 0:
            raise ValueError("srht JL requires n_embd to be a power of two")
        def hadamard(n):
            H = torch.tensor([[1.0]], device="cpu")
            size = 1
            while size < n:
                H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
                size *= 2
            return H
        D = torch.randint(0, 2, (old_embd,), generator=g, device="cpu", dtype=torch.float32) * 2 - 1
        H = hadamard(old_embd)
        idx = torch.randperm(old_embd, generator=g)[: args.out_embd]
        proj = H[idx] * D
        proj /= math.sqrt(args.out_embd)
    elif args.jl_type == "qr":
        console.print("[yellow]Generating orthogonal JL (QR) …")
        raw = torch.randn(args.out_embd, old_embd, generator=g, device="cpu")
        # QR on the *transpose* to avoid huge tall-skinny matrix when out_embd < old_embd
        q, _ = torch.linalg.qr(raw.T, mode="reduced")      # (old_embd, out_embd)
        proj = q.T                                         # (out_embd, old_embd)
        proj /= math.sqrt(args.out_embd)
    else:
        proj = sign_matrix(args.out_embd, old_embd, g, device="cpu")

    # ------------------------------------------------------------------ #
    # Optionally persist the projection matrix
    # ------------------------------------------------------------------ #
    if args.proj_out:
        console.print(f"[green]Saving projection matrix →[/] {args.proj_out}")
        torch.save(proj, args.proj_out)

    checkpoint.setdefault("config", {})["attn_variant"] = checkpoint.get(
        "config", {}
    ).get("attn_variant", "infinite")

    # gather mlp expansion sizes so we can preserve them regardless of new n_embd
    mlp_sizes = []

    # -------------------- progress bar over tensors -------------------- #
    total_fp = sum(1 for t in state_dict.values() if torch.is_floating_point(t))
    console.rule("[bold cyan]Projecting tensors")
    bar = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
    )
    tid = bar.add_task("JL-mapping", total=total_fp)

    with bar:
        for key, tensor in list(state_dict.items()):
            if not torch.is_floating_point(tensor):
                continue

            if key.endswith("mlp.c_fc.weight") and tensor.ndim == 2:
                mlp_sizes.append(tensor.shape[0])

            vertical = args.cproj_vertical and key.endswith("c_proj.weight")
            state_dict[key] = jl_project_tensor(tensor, proj, vertical_only=vertical)
            bar.advance(tid)

    if "model_args" in checkpoint:
        checkpoint["model_args"]["n_embd"] = args.out_embd
        if "n_embd_wte" in checkpoint["model_args"] and checkpoint["model_args"]["n_embd_wte"] == old_embd:
            checkpoint["model_args"]["n_embd_wte"] = args.out_embd
        if old_qk_dim is not None:
            checkpoint["model_args"]["n_qk_head_dim"] = old_qk_dim
        if old_v_dim is not None:
            checkpoint["model_args"]["n_v_head_dim"] = old_v_dim
        if mlp_sizes:
            checkpoint["model_args"]["mlp_size"] = mlp_sizes[0] if len(set(mlp_sizes)) == 1 else None
            if len(set(mlp_sizes)) > 1:
                checkpoint["model_args"]["mlp_size_layerlist"] = mlp_sizes

    if "config" in checkpoint:
        checkpoint["config"]["n_embd"] = args.out_embd
        if old_qk_dim is not None:
            checkpoint["config"]["n_qk_head_dim"] = old_qk_dim
        if old_v_dim is not None:
            checkpoint["config"]["n_v_head_dim"] = old_v_dim
        if mlp_sizes:
            checkpoint["config"]["mlp_size"] = mlp_sizes[0] if len(set(mlp_sizes)) == 1 else None
            if len(set(mlp_sizes)) > 1:
                checkpoint["config"]["mlp_size_layerlist"] = mlp_sizes

    # reset training progress so fine-tuning starts fresh
    checkpoint["iter_num"] = 0
    checkpoint["best_val_loss"] = 1e9
    checkpoint["best_iter"] = 0
    if args.dry_run:
        console.rule("[bold red]Dry-run complete -- nothing written[/]")
        console.print(f"Projected shapes preserved • {total_fp} tensors visited")
        return

    out_dir = args.out_dir or f"{args.ckpt_dir.rstrip('/').rstrip(os.sep)}_jl"
    os.makedirs(out_dir, exist_ok=True)

    console.rule("[bold cyan]Saving new checkpoint")
    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    if os.path.exists(meta_path):
        shutil.copy2(meta_path, os.path.join(out_dir, "meta.pkl"))

    console.print(
        f"[green]✔ All done[/] → {out_dir}  "
        f"([bold]{args.out_embd}[/]-dim, JL = {args.jl_type})  "
        f"at {datetime.datetime.now().isoformat(timespec='seconds')}"
    )


if __name__ == "__main__":
    main()

