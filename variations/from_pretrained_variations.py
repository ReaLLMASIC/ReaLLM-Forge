import torch

@classmethod
def from_pretrained_qwen(cls, config, model_type="Qwen/Qwen2-7B"):
    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"loading weights from pretrained gpt: {model_type}")

    model = cls(config)
    sd = model.state_dict()
    
    config_hf = AutoConfig.from_pretrained(model_type, trust_remote_code=True)
    model_hf = AutoModelForCausalLM.from_pretrained(model_type, config=config_hf, trust_remote_code=True)
    sd_hf = model_hf.state_dict()

    parameter_mapping = {}

    # Map the embeddings
    parameter_mapping['transformer.wte.weight'] = 'model.embed_tokens.weight'

    # Map the layers
    for i in range(config.n_layer):
        user_layer_prefix = f'transformer.h.{i}'
        hf_layer_prefix = f'model.layers.{i}'

        # Map attention weights
        parameter_mapping[f'{user_layer_prefix}.attn.c_attn_q.weight'] = f'{hf_layer_prefix}.self_attn.q_proj.weight'
        parameter_mapping[f'{user_layer_prefix}.attn.c_attn_q.bias'] = f'{hf_layer_prefix}.self_attn.q_proj.bias'

        parameter_mapping[f'{user_layer_prefix}.attn.c_attn_k.weight'] = f'{hf_layer_prefix}.self_attn.k_proj.weight'
        parameter_mapping[f'{user_layer_prefix}.attn.c_attn_k.bias'] = f'{hf_layer_prefix}.self_attn.k_proj.bias'

        parameter_mapping[f'{user_layer_prefix}.attn.c_attn_v.weight'] = f'{hf_layer_prefix}.self_attn.v_proj.weight'
        parameter_mapping[f'{user_layer_prefix}.attn.c_attn_v.bias'] = f'{hf_layer_prefix}.self_attn.v_proj.bias'

        parameter_mapping[f'{user_layer_prefix}.attn.c_proj.weight'] = f'{hf_layer_prefix}.self_attn.o_proj.weight'

        # Map layer normalization weights
        parameter_mapping[f'{user_layer_prefix}.ln_1.gain'] = f'{hf_layer_prefix}.input_layernorm.weight'

        parameter_mapping[f'{user_layer_prefix}.ln_2.gain'] = f'{hf_layer_prefix}.post_attention_layernorm.weight'

        assert config.mlp_variant == "swiglu"
        # Map for SwiGLU activation variant
        parameter_mapping[f'{user_layer_prefix}.mlp.c_fc_in1.weight'] = f'{hf_layer_prefix}.mlp.gate_proj.weight'

        parameter_mapping[f'{user_layer_prefix}.mlp.c_fc_in2.weight'] = f'{hf_layer_prefix}.mlp.up_proj.weight'

        parameter_mapping[f'{user_layer_prefix}.mlp.c_fc_out.weight'] = f'{hf_layer_prefix}.mlp.down_proj.weight'

    # Map final layer normalization weights
    parameter_mapping['transformer.ln_f.gain'] = 'model.norm.weight'

    # Map the output head
    parameter_mapping['lm_head.weight'] = 'lm_head.weight'

    # Load the parameters
    for model_param, hf_param in parameter_mapping.items():
        if model_param in sd and hf_param in sd_hf:
            if sd[model_param].shape == sd_hf[hf_param].shape:
                with torch.no_grad():
                    sd[model_param].copy_(sd_hf[hf_param])
            else:
                print(f"Shape mismatch for parameter {model_param}: "
                    f"model shape {sd[model_param].shape} vs "
                    f"pretrained shape {sd_hf[hf_param].shape}")
        else:
            print(f"Parameter {model_param} or {hf_param} not found in the respective models.")

    return model

@classmethod
def from_pretrained(cls, config, model_type):
    # assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    from transformers import GPT2LMHeadModel

    print(f"loading weights from pretrained gpt: {model_type}")

    # create a from-scratch initialized minGPT model
    model = cls(config)

    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

    # init a huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    # copy while ensuring all of the parameters are aligned and match in names and shapes
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
    transposed = ['attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
    # this means that we have to transpose these weights when we import them
    # NOTE: the assert below will fail because we split out the c_attn linears!
    # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for key in sd_keys_hf:
        if any(key.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            assert sd_hf[key].shape[::-1] == sd[key].shape
            with torch.no_grad():
                sd[key].copy_(sd_hf[key].t())
        elif key.endswith('attn.c_attn.weight') or key.endswith('attn.c_attn.bias'):
            # split into c_attn_q/k/v
            q, k, v  = sd_hf[key].t().split(config.n_embd, dim=0)
            q_key_str = key.replace("c_attn", "c_attn_q")
            k_key_str = key.replace("c_attn", "c_attn_k")
            v_key_str = key.replace("c_attn", "c_attn_v")
            sd[q_key_str] = q
            sd[k_key_str] = k
            sd[v_key_str] = v
        else:
            # vanilla copy over the other parameters
            print(key)
            if config.n_embd_wte:
                if key == "transformer.wte.weight":
                    continue
                if key == "lm_head.weight":
                    continue

            if not config.use_abs_pos_embeddings:
                if key == "transformer.wpe.weight":
                    continue

            assert sd_hf[key].shape == sd[key].shape
            with torch.no_grad():
                print(key)
                sd[key].copy_(sd_hf[key])

    return model