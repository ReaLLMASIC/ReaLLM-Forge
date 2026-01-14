import random, math, pickle
import os, time
from typing import Any, Dict, List
from search_space import HeteroSearchSpace, Individual
import hashlib, json, csv
from remote_trainer import RemoteTrainer

# -----------------------------
# Problem with proxy evaluation
# -----------------------------
class EvaluationResult:
    def __init__(self, objs: List[float], cons: List[float], aux: Dict[str, Any]):
        self.objs = objs
        self.cons = cons
        self.aux = aux

class Problem:
    def __init__(self, space: HeteroSearchSpace):
        self.space = space
        # now using 3 objectives: val_loss, energy_per_token, TTFT
        self.n_objs = 3
        self.mem_budget_bytes = 1_200_000_000  # ~1.2GB
        self.param_budget = 110_000_000
        self.latency_limit_ms = 180.0
        self._cache: Dict[str, EvaluationResult] = {}

    def hash(self, x: Individual) -> str:
        return hashlib.md5(json.dumps(x, sort_keys=True).encode()).hexdigest()

    def eval(self, x: Individual) -> EvaluationResult:
        key = self.hash(x)
        if key in self._cache:
            return self._cache[key]
        params = estimate_params_hetero(x)
        mem_bytes = estimate_mem_hetero(x)
        flops = estimate_flops_hetero(x)
        val_loss, throughput, e_per_token, ttft = proxy_measure(x, params, mem_bytes, flops)
        # constraints
        c1 = params - self.param_budget
        c2 = mem_bytes - self.mem_budget_bytes
        c3 = latency_from_tp(throughput) - self.latency_limit_ms
    # three objectives: minimize val_loss, minimize energy per token, minimize TTFT
        objs = [float(val_loss), float(e_per_token), float(ttft)]
        res = EvaluationResult(objs, [c1, c2, c3], {
            "params": params, "mem_bytes": mem_bytes, "FLOPs": flops,
            "val_loss": val_loss, "throughput": throughput, "energy/token": e_per_token, "TTFT": ttft,
            "globals": x["globals"]
        })
        self._cache[key] = res
        return res

class Population:
    # Holds individuals and their evaluations
    # initialized after evaluation 
    def __init__(self, individuals: List[Individual], evaluations: List[EvaluationResult] = None, search_space: HeteroSearchSpace = None):
        self.individuals: List[Individual] = []
        for ind in individuals:
            if isinstance(ind, Individual):
                self.individuals.append(ind)
            elif isinstance(ind, dict):
                self.individuals.append(Individual.from_dict(ind))
            else:
                raise TypeError(f"Expected Individual or dict, got {type(ind)}")
        self.evaluations: List[EvaluationResult] = evaluations or []
        self.offspring: List[Individual] = []
        self.offspring_evaluations: List[EvaluationResult] = []
        self.gen = 0

        self.search_space = search_space

        # parameter options
        self.n_population = 16
        self.n_offspring = 8
        self.tournament_k = 2  # tournament selection size
        self.mutation_rate = 0.1  # mutation rate for offspring
        self.crossover_rate = 0.9  # crossover rate for offspring

    def print_summary(self):
        """Print a formatted summary of the population."""
        print(f"\n=== Population Summary (Generation {self.gen}) ===")
        print(f"Population size: {len(self.individuals)}")
        if self.offspring:
            print(f"Offspring size: {len(self.offspring)}")
        
        if self.evaluations:
            print(f"Evaluations completed: {len(self.evaluations)}")
            # Show objective statistics
            objs = [ev.objs for ev in self.evaluations]
            if objs:
                val_losses = [obj[0] for obj in objs]
                energy_per_token = [obj[1] for obj in objs]
                ttft = [obj[2] for obj in objs]
                
                print(f"\nObjective Statistics:")
                print(f"  Validation Loss: {min(val_losses):.3f} - {max(val_losses):.3f} (avg: {sum(val_losses)/len(val_losses):.3f})")
                print(f"  Energy/Token:    {min(energy_per_token):.3f} - {max(energy_per_token):.3f} (avg: {sum(energy_per_token)/len(energy_per_token):.3f})")
                print(f"  TTFT:           {min(ttft):.3f} - {max(ttft):.3f} (avg: {sum(ttft)/len(ttft):.3f})")
                
                # Show constraint violations
                cons = [ev.cons for ev in self.evaluations]
                if cons:
                    param_violations = sum(1 for c in cons if c[0] > 0)
                    mem_violations = sum(1 for c in cons if c[1] > 0)
                    # latency_violations = sum(1 for c in cons if c[2] > 0)
                    print(f"\nConstraint Violations:")
                    print(f"  Parameter budget: {param_violations}/{len(cons)} individuals")
                    print(f"  Memory budget:    {mem_violations}/{len(cons)} individuals")
                    # print(f"  Latency limit:    {latency_violations}/{len(cons)} individuals")
        else:
            print("No evaluations completed yet")
        
        print("=" * 50)

    def print_details(self):
        """Print detailed information of each individual and its evaluation."""
        for i, (ind, ev) in enumerate(zip(self.individuals, self.evaluations or [])):
            print(f"\n--- Individual {i+1} ---")
            indiv: Individual = ind
            indiv.print_individual()
            if ev:
                print(f"Objectives: {ev.objs}")
                print(f"Constraints: {ev.cons}")
                print(f"Auxiliary Info: {ev.aux}")
            else:
                print("No evaluation result available.")
        
        print("\n------------------------------------------")
        for i, ind in enumerate(self.offspring):
            print(f"\n--- Offspring Individual {i+1} ---")
            ind.print_individual()
            if self.offspring_evaluations and i < len(self.offspring_evaluations):
                ev = self.offspring_evaluations[i]
                print(f"Objectives: {ev.objs}")
                print(f"Constraints: {ev.cons}")
                print(f"Auxiliary Info: {ev.aux}")
            else:
                print("No evaluation result available.")

        print("\n" + "=" * 50)

    def __str__(self):
        """String representation of the population."""
        return f"Population(gen={self.gen}, size={len(self.individuals)}, evaluated={len(self.evaluations) if self.evaluations else 0})"

    def delete_duplicates(self):
        unique = {}
        for ind in self.individuals:
            key = json.dumps(ind, sort_keys=True)
            if key not in unique:
                unique[key] = ind
        self.individuals = list(unique.values())

    def fast_non_dominated_sort(self, objs: List[List[float]] = None, cons: List[List[float]] = None) -> List[List[int]]:
        """Perform non-dominated sorting and return a list of fronts (each front is a list of indices).

        If objs/cons are not provided, they will be derived from self.evaluations.
        - objs: List of objective vectors, each a list of floats (minimization).
        - cons: List of constraint vectors, each a list of floats (<= 0 is feasible).
        """
        # Derive from current evaluations if not explicitly given
        if objs is None or cons is None:
            if not self.evaluations:
                return []
            objs = [e.objs for e in self.evaluations]
            cons = [e.cons for e in self.evaluations]

        N = len(objs)
        S = [[] for _ in range(N)]
        n = [0] * N
        rank = [None] * N
        F: List[List[int]] = [[]]
        for p in range(N):
            for q in range(N):
                if p == q:
                    continue
                d = dominates(objs[p], cons[p], objs[q], cons[q])
                if d == 1:
                    S[p].append(q)
                elif d == -1:
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                F[0].append(p)
        i = 0
        while F[i]:
            Q: List[int] = []
            for p in F[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i += 1
            F.append(Q)
        return F[:-1]

    def reorder_by_non_domination(self) -> None:
        """Reorder population in-place by Pareto fronts and crowding distance.

        Returns the index permutation applied (indices into the previous order).

        Ordering rule:
        1) Sort individuals into non-dominated fronts (F0, F1, ...)
        2) Within each front, sort by crowding distance descending (diversity preference)
        The final order is F0 (by CD), then F1 (by CD), etc.
        """
        if not self.evaluations:
            return []
        objs = [e.objs for e in self.evaluations]
        cons = [e.cons for e in self.evaluations]
        fronts = self.fast_non_dominated_sort(objs, cons)
        order: List[int] = []
        for front in fronts:
            if not front:
                continue
            cd = crowding_distance(front, objs)
            order.extend(sorted(front, key=lambda i: cd[i], reverse=True))
        # Apply permutation to individuals and evaluations
        self.individuals = [self.individuals[i] for i in order]
        self.evaluations = [self.evaluations[i] for i in order]
        print(f"Reordered individuals and evaluations by non-domination: {order}")
        return
    
    def to_yaml(self, save_path: str = None) -> str:
        """Convert population to YAML format for training experiments.
        Only includes active layers based on layer_mask.
        """
        yaml_lines = ["# Example YAML configuration file for training experiments"]
        yaml_lines.append("# Generated from NSGA-II population")
        yaml_lines.append("# Note: n_layers is automatically determined from the length of n_head_layerlist")
        yaml_lines.append("")

        for i, individual in enumerate(self.individuals if self.gen == 0 else self.offspring):
            g = individual["globals"]
            layers = individual["layers"]
            mask = g.get("layer_mask", [True] * len(layers))
            
            # Get active layers only
            active_indices = [j for j, active in enumerate(mask) if active]
            
            if not active_indices:  # Skip if no active layers
                continue

            # Format YAML entry
            yaml_lines.append(f"- idx: {i+1}")
            
            # add the configs in global settings
            for key, value in g.items():
                if key != "layer_mask":  # Exclude layer_mask from globals
                    yaml_lines.append(f"  {key}: {value}")

            # Build lists for active layers
            for key, _ in layers[0].items():
                layerlist = []
                arg = f"{key}_layerlist"
                for j in active_indices:
                    if j < len(layers):
                        layer = layers[j]
                        layerlist.append(layer.get(key))
                yaml_lines.append(f"  {arg}: {layerlist}")

            yaml_lines.append(f"# n_layers: {len(active_indices)}")
            
            yaml_lines.append("")

        yaml_output = "\n".join(yaml_lines)
        file_name = f"{save_path or 'population'}/gen{self.gen}.yaml"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as f:
            f.write(yaml_output)
        
        return file_name

    def save_checkpoint(self, path: str) -> str:
        """Save a checkpoint of the population to JSON.

        Contents: gen (int), individuals, evaluations, offspring, offspring_evaluations,
        search_space config, and all population parameters.
        Writes atomically via a temporary file then rename.
        Returns the final path.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "gen": int(self.gen),
            "individuals": self.individuals,
            "evaluations": None if self.evaluations is None else [
                {"objs": ev.objs, "cons": ev.cons, "aux": ev.aux} for ev in self.evaluations
            ],
            "offspring": self.offspring,
            "offspring_evaluations": None if self.offspring_evaluations is None else [
                {"objs": ev.objs, "cons": ev.cons, "aux": ev.aux} for ev in self.offspring_evaluations
            ],
            # Population parameters
            "n_offspring": self.n_offspring,
            "tournament_k": self.tournament_k,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            # Search space configuration (if available)
            "search_space_config": None if self.search_space is None else {
                "L_max": getattr(self.search_space, 'L_max', None),
                "d_model_choices": getattr(self.search_space, 'd_model_choices', None),
                "block_size_choices": getattr(self.search_space, 'block_size_choices', None),
                # Add other search space attributes as needed
            }
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
        return path

    def save_checkpoint_pkl(self, path: str) -> None:
        """Save a checkpoint of the population to a pickle file.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return

    @staticmethod
    def load_checkpoint(path: str, from_pkl=True) -> "Population":
        """Load a Population from a checkpoint created by save_checkpoint.

        Note: EvaluationResult objects are reconstructed from stored dicts.
        Search space must be re-initialized separately if needed for operations.
        """
        if from_pkl:
            with open(path, "rb") as f:
                pop = pickle.load(f)
            if not isinstance(pop, Population):
                raise TypeError(f"Loaded object is not a Population, got {type(pop)}")
            return pop
        
        with open(path, "r") as f:
            data = json.load(f)
        
        # Load basic data
        individuals = data.get("individuals", [])
        
        # Load evaluations
        evals_raw = data.get("evaluations")
        evaluations = None
        if evals_raw is not None:
            evaluations = [EvaluationResult(er["objs"], er["cons"], er.get("aux", {})) for er in evals_raw]
        
        # Load offspring
        offspring = data.get("offspring", [])
        
        # Load offspring evaluations
        offspring_evals_raw = data.get("offspring_evaluations")
        offspring_evaluations = None
        if offspring_evals_raw is not None:
            offspring_evaluations = [EvaluationResult(er["objs"], er["cons"], er.get("aux", {})) for er in offspring_evals_raw]
        
        # Create population
        pop = Population(individuals, evaluations)
        pop.gen = int(data.get("gen", 0))
        pop.offspring = offspring
        pop.offspring_evaluations = offspring_evaluations if offspring_evaluations is not None else []
        
        # Restore population parameters
        pop.n_offspring = data.get("n_offspring", 32)
        pop.tournament_k = data.get("tournament_k", 2)
        pop.mutation_rate = data.get("mutation_rate", 0.1)
        pop.crossover_rate = data.get("crossover_rate", 0.9)
        
        # Note: search_space is not restored as it requires re-initialization
        # User should call pop.search_space = HeteroSearchSpace(...) after loading if needed
        
        return pop

    def sw_eval(self, hosts: List[str], user: str, key_filename: str, run_dir_name: str = "default", max_iters: int = 10000, conda_env: str = "reallmforge") -> None:
        # send the training work to worker nodes and wait for results
        train_yaml_path = self.to_yaml(save_path="train")
        trainer = RemoteTrainer(hosts=hosts, user=user, key_filename=key_filename)
        trainer.submit_job(path_to_yaml=train_yaml_path, remote_work_dir=f"/home/{user}/Evo_GPT", dir_name=run_dir_name, max_iters=max_iters, conda_env=conda_env)
        trainer.wait_for_all(poll_interval=300, timeout=72000, verbose=True)
        data_csv = trainer.fetch_results(local_dir="train", gen=self.gen)
        # read the csv and populate self.evaluations
        # load the csv file's second column as a list of floats
        sw_data = load_csv_with_idx_lookup(data_csv)
        print (f"Loaded {len(sw_data)} results from {data_csv}")

        if self.gen == 0:
            self.evaluations = []
        else:
            self.offspring_evaluations = []

        for i, ind in enumerate(self.individuals if self.gen == 0 else self.offspring):
            idx = i + 1  # CSV idx starts from 1
            if idx in sw_data:
                val_loss = sw_data[idx]
                # reconstruct evaluation result
                # params = estimate_params_hetero_infi(ind)
                params = ind.estimate_params()
                mem_bytes = ind.estimate_mem_access()
                flops = ind.estimate_flops()
                _, e_per_token, ttft = proxy_measure(ind, params, mem_bytes, flops)
                c1 = params - 1_000_000_000
                c2 = val_loss - 3.0
                # c3 = latency_from_tp(throughput) - 180.0
                objs = [float(val_loss), float(e_per_token), float(ttft)]
                eval_res = EvaluationResult(objs, [c1, c2], {
                    "params": params, "mem_bytes": mem_bytes, "FLOPs": flops
                })
                if self.gen == 0:
                    self.evaluations.append(eval_res)
                else:
                    self.offspring_evaluations.append(eval_res)
            else:
                print(f"Warning: No result for individual idx {idx} in CSV")

            ind.print_individual()
            print(f"gen {self.gen} individual {idx}: val_loss={val_loss:.3f}, energy/token={e_per_token:.3f}, TTFT={ttft:.3f}, params={params/1e6:.1f}M, mem={mem_bytes/1e9:.2f}GB")
        return

    def generate_offspring(self) -> None:
        """Generate offspring via tournament selection and mutation."""
        if self.evaluations is None or not self.evaluations:
            raise ValueError("Cannot generate offspring without evaluations.")
        if self.search_space is None:
            raise ValueError("Search space is not defined for mutation.")
        search_space = self.search_space
        offspring = []
        for _ in range(self.n_offspring):
            p1_idx = tournament_select(self.individuals, self.evaluations, k=self.tournament_k)
            p2_idx = tournament_select(self.individuals, self.evaluations, k=self.tournament_k)
            parent1, _ = search_space.crossover(self.individuals[p1_idx], self.individuals[p2_idx], self.crossover_rate)
            child1 = self.search_space.mutate(parent1, self.mutation_rate)
            # child2 = self.search_space.mutate(parent2, self.mutation_rate)
            print("Generated offspring:")
            child1.print_individual()
            offspring.append(child1)

        self.offspring = offspring
        self.offspring_evaluations = []
        self.gen += 1
        print(f"Generated {self.n_offspring} offspring for generation {self.gen}")
        return

    def update_elimination(self, verbose: bool = False) -> None:
        if self.offspring_evaluations is None or not self.offspring_evaluations:
            raise ValueError("Cannot update elimination without offspring evaluations.")

        # append offspring to current population
        self.individuals.extend(self.offspring)
        self.evaluations.extend(self.offspring_evaluations)

        # Clear the offspring lists for the next generation
        self.offspring = []
        self.offspring_evaluations = []

        # Reorder by non-domination and keep the best individuals
        self.reorder_by_non_domination()
        if len(self.individuals) > self.n_population:
            print(f"Eliminating {len(self.individuals) - self.n_population} individuals to maintain population size {self.n_population}.")
            self.individuals = self.individuals[:self.n_population]
            self.evaluations = self.evaluations[:self.n_population]
        else:
            print(f"Population size {len(self.individuals)} is within limit {self.n_population}, no elimination needed.")

        if verbose:
            print(f"After elimination, population size: {len(self.individuals)}")
            for i, (ind, ev) in enumerate(zip(self.individuals, self.evaluations)):
                print(f"Individual {i+1}: Objs={ev.objs}, Cons={ev.cons}, Aux={ev.aux}")
        return

# -----------------------------
# CSV loading utility
# -----------------------------
def load_csv_with_idx_lookup(filepath):
    """Load CSV and return a dict for idx-based lookup."""
    data = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['#idx'])
            best_val_loss = float(row[' best_val_loss'])
            data[idx] = best_val_loss
    return data

def estimate_params_hetero(x: Individual):
    g = x["globals"]; d = g["d_model"]
    total = 0.0
    mask = x["globals"].get("layer_mask", [True]*len(x["layers"]))
    indices = [i for i,active in enumerate(mask) if active]
    for i in indices:
        li = x["layers"][i]
        h = max(1, li["n_heads"])
        r = li["mlp_ratio"]
        total += (2.0 + r) * d * d + 0.03 * h * d * d
    return int(total)

def estimate_flops_hetero(x: Individual):
    g = x["globals"]; d = g["d_model"]; seq = g["block_size"]
    cost = 0.0
    mask = x["globals"].get("layer_mask", [True]*len(x["layers"]))
    indices = [i for i,active in enumerate(mask) if active]
    for i in indices:
        li = x["layers"][i]
        attn = li["attn_type"]
        attn_cost = {"scaled_dot": d*seq,
                     "gqa": d*seq/2,
                     "mha": d*math.log2(max(2, seq)),
                     "flash": d*seq*0.7}.get(attn, d*seq)
        cost += (2*d*d + attn_cost) + li["mlp_ratio"]*d*d
    return cost

def estimate_mem_hetero(x: Individual):
    params = estimate_params_hetero(x)
    bytes_per_param = max(1, x["globals"]["quant_bits"] // 8)
    seq = x["globals"]["block_size"]; d = x["globals"]["d_model"]
    # KV cache proxy (rough): 4 * seq * d bytes
    kv = 4 * seq * d
    return int(params*bytes_per_param + kv)

def latency_from_tp(tp_tok_s: float) -> float:
    if tp_tok_s <= 1e-9: return 1e9
    return 1000.0 / tp_tok_s

def proxy_measure(x: Individual, params, mem_bytes, flops):
    scale = 1e-9
    # throughput decreases with flops and params (rough)
    # throughput = max(0.5, 30_000_000 / (flops*scale + 3.0) / (1.0 + params/150e6))
    # TTFT depends on params and whether using flash/linear
    mask = x["globals"].get("layer_mask", [True]*len(x["layers"]))
    L = sum(1 for v in mask if v)
    ttft = params/1e6  #ms  scale with size
    # energy per token depends on flops and memory
    e_per_token = flops/1e3
    # val_loss improves with capacity but worsens with aggressive quant + sparsity
    val_loss = 99
    return val_loss, e_per_token, ttft

# -----------------------------
# NSGA-II core
# -----------------------------
def dominates(o1, c1, o2, c2):
    feas1 = all(c <= 0 for c in c1)
    feas2 = all(c <= 0 for c in c2)
    if feas1 and not feas2: return 1
    if feas2 and not feas1: return -1
    if feas1 and feas2:
        better = False; worse = False
        for a,b in zip(o1,o2):
            if a < b - 1e-12: better = True
            elif a > b + 1e-12: worse = True
        if better and not worse: return 1
        if worse and not better: return -1
        return 0
    v1 = sum(max(0.0, c) for c in c1)
    v2 = sum(max(0.0, c) for c in c2)
    if v1 < v2 - 1e-12: return 1
    if v1 > v2 + 1e-12: return -1
    return 0

def fast_non_dominated_sort(objs, cons):
    N = len(objs)
    S = [[] for _ in range(N)]
    n = [0]*N
    rank = [None]*N
    F = [[]]
    for p in range(N):
        for q in range(N):
            if p==q: continue
            d = dominates(objs[p], cons[p], objs[q], cons[q])
            if d == 1: S[p].append(q)
            elif d == -1: n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            F[0].append(p)
    i = 0
    while F[i]:
        Q = []
        for p in F[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        F.append(Q)
    return F[:-1]

def crowding_distance(front, objs):
    if not front: return {}
    m = len(objs[0])
    dist = {i: 0.0 for i in front}
    for k in range(m):
        front_sorted = sorted(front, key=lambda i: objs[i][k])
        fmin = objs[front_sorted[0]][k]
        fmax = objs[front_sorted[-1]][k]
        dist[front_sorted[0]] = float('inf')
        dist[front_sorted[-1]] = float('inf')
        denom = (fmax - fmin) if abs(fmax - fmin) > 1e-12 else 1.0
        for i in range(1, len(front_sorted)-1):
            prev = objs[front_sorted[i-1]][k]
            nxt  = objs[front_sorted[i+1]][k]
            dist[front_sorted[i]] += (nxt - prev)/denom
    return dist

def tournament_select(pop, evals, k=2):
    i = random.randrange(len(pop))
    for _ in range(k-1):
        j = random.randrange(len(pop))
        if j == i: continue
        d = dominates(evals[i].objs, evals[i].cons, evals[j].objs, evals[j].cons)
        if d == -1 or (d == 0 and random.random() < 0.5):
            i = j
    return i

def nsga2(problem, pop_size=32, n_gen=6, seed=42, log_fn=None, log_every: int = 1, verbose: bool = False):
    random.seed(seed)
    space = problem.space
    pop = [space.sample() for _ in range(pop_size)]
    evals = [problem.eval(ind) for ind in pop]
    history = []
    for gen in range(n_gen):
        pop_before_len = len(pop)
        offspring = []
        while len(offspring) < pop_size:
            p1 = pop[tournament_select(pop, evals)]
            p2 = pop[tournament_select(pop, evals)]
            c1, c2 = space.crossover(p1, p2)
            c1 = space.mutate(c1)
            c2 = space.mutate(c2)
            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)
        off_evals = [problem.eval(ind) for ind in offspring]
        union = pop + offspring
        union_evals = evals + off_evals
        objs = [e.objs for e in union_evals]
        cons = [e.cons for e in union_evals]
        fronts = fast_non_dominated_sort(objs, cons)
        new_pop = []
        new_evals = []
        selected_union_idx = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                for idx in front:
                    new_pop.append(union[idx]); new_evals.append(union_evals[idx]); selected_union_idx.append(idx)
            else:
                cd = crowding_distance(front, objs)
                sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                for idx in sorted_front[:pop_size - len(new_pop)]:
                    new_pop.append(union[idx]); new_evals.append(union_evals[idx]); selected_union_idx.append(idx)
                break
        pop, evals = new_pop, new_evals
        # compute current-population Pareto summary (not the union)
        cur_objs = [e.objs for e in evals]
        cur_cons = [e.cons for e in evals]
        cur_fronts = fast_non_dominated_sort(cur_objs, cur_cons)
        bf = cur_fronts[0] if cur_fronts and cur_fronts[0] else []
        bf_objs = [cur_objs[i] for i in bf] if bf else []
        if bf_objs:
            avg = tuple(round(sum(col)/len(col), 4) for col in zip(*bf_objs))
        else:
            avg = tuple()
        gen_log = {"gen": gen+1, "pareto_size": len(bf), "avg_objs": avg}
        history.append(gen_log)
        # optional external logging callback
        if log_fn is not None and (gen + 1) % max(1, log_every) == 0:
            try:
                log_state = {
                    "ts": time.time(),
                    "gen": gen + 1,
                    "pareto_size": len(bf),
                    "avg_objs": avg,
                }
                if verbose:
                    # Include heavy details only when verbose is enabled
                    pareto_set = set(bf)
                    log_state.update({
                        "fronts": cur_fronts,
                        "objs": cur_objs,
                        "cons": cur_cons,
                        "population_size": len(pop),
                        "offspring_generated": len(offspring),
                    })
                    population = []
                    for i_pop in range(len(pop)):
                        ev = evals[i_pop]
                        population.append({
                            "x": pop[i_pop],
                            "objs": ev.objs,
                            "cons": ev.cons,
                            "aux": ev.aux,
                            "on_pareto": i_pop in pareto_set,
                        })
                    offspring_logs = []
                    base = pop_before_len
                    selected_set = set(selected_union_idx)
                    selected_offspring_count = 0
                    for k in range(len(offspring)):
                        e = off_evals[k]
                        sel = (base + k) in selected_set
                        if sel:
                            selected_offspring_count += 1
                        offspring_logs.append({
                            "x": offspring[k],
                            "objs": e.objs,
                            "cons": e.cons,
                            "aux": e.aux,
                            "selected": sel,
                        })
                    log_state["population"] = population
                    log_state["offspring"] = offspring_logs
                    log_state["offspring_selected"] = selected_offspring_count
                log_fn(log_state)
            except Exception:
                # keep search running even if logging fails
                pass
    return pop, evals, history


def make_jsonl_logger(path: str):
    """Return a logger callable that appends one JSON object per generation to `path`.

    Each record at minimum contains: ts, gen, pareto_size, avg_objs.
    When nsga2(verbose=True), records may also include: fronts, objs, cons,
    population, offspring (with selection flags).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _log_fn(state: Dict[str, Any]):
        # ensure JSON serializable content; state is constructed from primitives
        with open(path, "a") as f:
            f.write(json.dumps(state) + "\n")

    return _log_fn


def make_json_list_logger(path: str):
    """Return a logger that stores all generation states in memory and writes a single JSON array.

    Use this when you prefer one compact `.json` artifact over a streaming `.jsonl` log.
    NOTE: Holds all records until process end; for very long runs prefer JSONL.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    buffer = []

    def _log_fn(state: Dict[str, Any]):
        buffer.append(state)
        # Overwrite file each time so you can tail partial progress safely
        try:
            with open(path, "w") as f:
                json.dump(buffer, f, indent=2)
        except Exception:
            pass

    return _log_fn

