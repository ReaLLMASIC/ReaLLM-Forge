import torch
import numpy as np

def direct_init(config):
    """
    Minimal example of 'direct' hypercube initialization.
    For demonstration, we'll just create 2^n_embd corners
    and then pick the first `vocab_size`.
    """
    # extract config variables
    vocab_size=config.vocab_size
    n_embd=config.n_embd
    scale=config.init_scale

    n_corners = 2 ** n_embd
    n_corners = min(n_corners, vocab_size)

    if vocab_size > n_corners:
        raise ValueError(
            f"Not enough corners (2^{n_embd}={n_corners}) for vocab_size={vocab_size} in 'direct' mode."
        )
    corners = torch.zeros((n_corners, n_embd))
    for i in range(n_corners):
        for d in range(n_embd):
            corners[i, d] = (i >> d) & 1
    return corners[:vocab_size, :] * scale

def one_hot_init(config):
    """
    Create a one-hot embedding matrix of shape [vocab_size, n_embd].
    We assert n_embd >= vocab_size so that each row can have exactly one 1
    in a distinct column.
    """

    # extract config variables
    vocab_size=config.vocab_size
    n_embd=config.n_embd
    scale=config.init_scale

    # init
    if n_embd < vocab_size:
        raise ValueError("For 'one-hot' init, n_embd must be >= vocab_size.")
    weight = torch.zeros((vocab_size, n_embd))
    for i in range(vocab_size):
        weight[i, i] = scale
    return weight

def numpy_import_init(config):
    """
    Loads a pre-trained embedding matrix from a NumPy file.
    The file_path will be handled by the GPT class.
    """

    # extract config variables
    vocab_size=config.vocab_size
    n_embd=config.n_embd
    scale=config.init_scale
    file_path = config.init_wte_npy

    try:
        embedding_data = np.load(file_path)
        embedding_tensor = torch.from_numpy(embedding_data).float()
        if embedding_tensor.shape != (vocab_size, n_embd):
            raise ValueError(f"Numpy embedding shape {embedding_tensor.shape} does not match expected shape ({vocab_size}, {n_embd}).")
        return embedding_tensor * scale
    except FileNotFoundError:
        raise FileNotFoundError(f"NumPy embedding file not found at {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading NumPy embedding file: {e}")

def random_hypercube_norm(config):
    """Random +/-1 hypercube corners normalized and scaled."""
    vocab_size = config.vocab_size
    n_embd = config.n_embd
    scale = config.init_scale

    weight = torch.randint(0, 2, (vocab_size, n_embd), dtype=torch.float32)
    weight = weight * 2 - 1  # convert {0,1} -> {-1,1}
    weight = weight / weight.norm(dim=1, keepdim=True)
    return weight * scale


def hypersphere_angle_init(config):
    """Random angles mapped onto a hypersphere surface."""
    vocab_size = config.vocab_size
    n_embd = config.n_embd
    radius = getattr(config, "init_radius", 1.0)

    angles = torch.rand(vocab_size, n_embd) * (2 * np.pi)
    vec = torch.cos(angles)
    vec = vec / vec.norm(dim=1, keepdim=True)
    return vec * radius


def unique_hypercube_norm(config):
    """Random hypercube corners ensuring no opposite pairs."""
    vocab_size = config.vocab_size
    n_embd = config.n_embd
    scale = config.init_scale

    max_unique = 2 ** n_embd // 2
    if vocab_size > max_unique:
        raise ValueError("Not enough unique corners without opposites.")

    vectors = []
    seen = set()
    while len(vectors) < vocab_size:
        vec = torch.randint(0, 2, (n_embd,), dtype=torch.float32)
        vec = vec * 2 - 1
        tup = tuple(vec.tolist())
        if tup in seen or tuple((-vec).tolist()) in seen:
            continue
        seen.add(tup)
        vectors.append(vec)

    weight = torch.stack(vectors)
    weight = weight / weight.norm(dim=1, keepdim=True)
    return weight * scale


def gaussian_norm_range_init(config):
    """Gaussian init with vector norm constraints."""
    vocab_size = config.vocab_size
    n_embd = config.n_embd
    mean = config.embedding_mean_init
    std = config.embedding_std_init
    min_norm = getattr(config, "gaussian_min_norm", 0.0)
    max_norm = getattr(config, "gaussian_max_norm", float("inf"))

    weight = torch.empty((vocab_size, n_embd), dtype=torch.float32)
    i = 0
    while i < vocab_size:
        vec = torch.randn(n_embd) * std + mean
        norm = vec.norm().item()
        if norm < min_norm or norm > max_norm:
            continue
        weight[i] = vec
        i += 1
    return weight


init_dictionary = {
    "gaussian": None,  # default Gaussian
    "hypercube": direct_init,
    "onehot": one_hot_init,
    "numpy_import": numpy_import_init,
    "rand_hypercube": random_hypercube_norm,
    "angle_hypersphere": hypersphere_angle_init,
    "unique_hypercube": unique_hypercube_norm,
    "gaussian_norm_range": gaussian_norm_range_init,
}

