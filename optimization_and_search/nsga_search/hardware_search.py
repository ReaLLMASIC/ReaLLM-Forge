from search_space import Individual, HeteroSearchSpace
from typing import List, Dict, Any, Tuple

class Hardware_encoding:
    def __init__(self, n_col: int, n_mac: int):
        # deterministic encoding of hardware configuration
        self.n_col = n_col
        self.n_mac = n_mac

        # derived attributes
        self.n_row = None
        self.n_pe = None
        self.wmem_depth = None
        self.kvcache_depth = None

        self.ac_wmem_depth = None
        self.ac_kvcache_depth = None

    def is_feasible(self, ind: Individual) -> bool:
        # Check if the hardware configuration is feasible for the given individual
        n_embd = ind["globals"]["d_model"]
        # Use a mask falling back to the number of layers if L_max isn't defined
        layers = ind["layers"]
        mask = list(ind["globals"].get("layer_mask", [True]*len(layers)))
        n_heads_list = [layer["n_heads"] if mask[i] else 0 for i, layer in enumerate(layers)]
        n_heads_max = max(n_heads_list)
        n_heads = n_heads_max
        
        block_size = ind["globals"]["block_size"]
        head_dim = n_embd // n_heads
        if n_embd % n_heads != 0:
            return False
        if n_embd % self.n_col != 0:
            return False
        core_dim = n_embd // self.n_col
        if core_dim % self.n_mac != 0:
            return False
        if block_size % self.n_col != 0:
            return False

        self.n_row = n_heads
        self.kvcache_depth = int(2 * n_embd * block_size / self.n_mac / self.n_col / self.n_row)
        self.n_pe = self.n_col * self.n_row
        self.wmem_depth = int(4 * n_embd * n_embd / self.n_row / self.n_col / self.n_mac)

        if self.wmem_depth > 8192 or self.kvcache_depth > 8192:
            return False

        self.ac_kvcache_depth = sram_depth_round_up(self.kvcache_depth)
        self.ac_wmem_depth = sram_depth_round_up(self.wmem_depth)
        return True

    def get_TTFT_in_cycle(self, ind: Individual, seq_len: int = 256) -> float:
        
        n_embd = ind["globals"]["d_model"]
        gbus_width = self.n_mac * 8
        n_heads = self.n_row
        n_cols = self.n_col
        block_size = ind["globals"]["block_size"]

        layers = ind["layers"]
        mask = list(ind["globals"].get("layer_mask", [True]*len(layers)))
        ttft_cycles = 0
        for i, layer in enumerate(ind["layers"]):
            if mask[i] == False:
                continue
            layer_cycle = 0
            mlp_ratio = layer["mlp_ratio"]
            mlp_size = n_embd * mlp_ratio
            layer_cycle += (4 * n_embd * n_embd + 2 * mlp_size * n_embd) / gbus_width  # load weights on chip

            layer_cycle += (4 * n_embd * n_embd * seq_len + 2 * block_size * block_size * n_embd + 2 * mlp_size * n_embd * seq_len) / (n_heads * n_cols * self.n_mac)  # MAC operations

            # v-link latency penalty
            if (layer["n_heads"] < n_heads) :
                n_groups = n_heads / layer["n_heads"]

                layer_cycle += (n_groups - 1) * (n_embd / n_heads) / gbus_width  # assuming perfect interleaving
                
                # mlp ratio penalty
                layer_cycle += (mlp_ratio - 2) * n_embd / gbus_width  # assuming perfect interleaving

            # add residual delay
            layer_cycle += 2 * n_embd * seq_len / (self.n_mac * n_heads * n_cols)  # load and store residual

            ttft_cycles += layer_cycle

        return ttft_cycles

    def get_token_energy_in_cycle(self, ind: Individual, seq_len: int = 256) -> float:
        n_embd = ind["globals"]["d_model"]
        gbus_width = self.n_mac * 8
        n_heads = self.n_row
        n_cols = self.n_col
        block_size = ind["globals"]["block_size"]

        layers = ind["layers"]
        mask = list(ind["globals"].get("layer_mask", [True]*len(layers)))
        ttft_cycles = 0
        for i, layer in enumerate(ind["layers"]):
            if mask[i] == False:
                continue
            layer_cycle = 0
            mlp_ratio = layer["mlp_ratio"]
            mlp_size = n_embd * mlp_ratio
            layer_cycle += 0.2 * (4 * n_embd * n_embd + 2 * mlp_size * n_embd) / gbus_width  # load weights on chip

            layer_cycle += (4 * n_embd * n_embd + 2 * block_size * block_size * n_embd + 2 * mlp_size * n_embd) / (n_heads * n_cols * self.n_mac)  # MAC operations

            # v-link latency penalty
            if (layer["n_heads"] < n_heads) :
                n_groups = n_heads / layer["n_heads"]

                layer_cycle += 0.2 * (n_groups - 1) * (n_embd / n_heads) / gbus_width  # assuming perfect interleaving
                
                # mlp ratio penalty
                layer_cycle += 0.2 * (mlp_ratio - 2) * n_embd / gbus_width  # assuming perfect interleaving

            # add residual delay
            layer_cycle += 2 * n_embd * seq_len / (self.n_mac * n_heads * n_cols)  # load and store residual

            ttft_cycles += layer_cycle

        return ttft_cycles

# Hardware stat dataclass
class HardwareStat:
    def __init__(self, params_m: float, mem_gb: float, energy_per_token: float, ttft: float):
        self.energy_per_token = energy_per_token
        self.ttft = ttft
        self.n_sram_access = None
        self.sram_rb = None
        self.sram_wb = None

# def evaluate_hardware(ind: Individual) -> Dict[str, Any]:
    # First map the individual to feasiable hardware configuration

def sram_depth_round_up(depth: int) -> int:
    # Round up to nearest power of 2
    if depth <= 128:
        return 128
    elif depth <= 1024:
        # round up to nearest 256
        return int((depth + 255) // 256 * 256)
    elif depth <= 4096:
        # round up to nearest 512
        return int((depth + 511) // 512 * 512)  
    else:
        # round up to nearest 1024
        return int((depth + 1023) // 1024 * 1024)

if __name__ == "__main__":
    init_population_size = 32
    max_n_layer = 10
    search_space = HeteroSearchSpace(L_max=max_n_layer)
    individuals = [search_space.sample() for _ in range(init_population_size)]

    for ind in individuals:
        for col in range(1, 33):  # n_col from 1 to 32:
            for mac in [4, 8, 16, 32]:
                hw = Hardware_encoding(n_col=col, n_mac=mac)
                if hw.is_feasible(ind):
                    print(f"Individual is feasible on hardware {hw.n_col}x{hw.n_row} with {hw.n_mac} MACs")
                    print(f"Wmem depth: {hw.wmem_depth} (actual {hw.ac_wmem_depth}), KVCache depth: {hw.kvcache_depth} (actual {hw.ac_kvcache_depth})")
                    energy = hw.get_token_energy_in_cycle(ind)
                    ttft = hw.get_TTFT_in_cycle(ind)
                    print(f"TTFT in cycle: {ttft}")
                    print(f"Token energy in cycle: {energy}")
                # else:
                    # print(f"Individual is NOT feasible on hardware {hw.n_col} with {hw.n_mac} MACs")

        print("****************************************")



    
