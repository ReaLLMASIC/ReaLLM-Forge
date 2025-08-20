# 🧬 Adding Variations

This guide provides a step-by-step walkthrough for adding new architectural variations to the ReaLLM-Forge model. The repository is designed to be highly modular, making it easy to experiment with new ideas.

## 1. Understand the Variation Dictionaries

The key to the modularity of the model is the use of "variation dictionaries." These are Python dictionaries that map a string name to a specific implementation of a module (e.g., an attention mechanism, an MLP layer, a normalization layer).

You can find these dictionaries in the `variations/` directory. For example, `variations/attention_variations.py` contains the `attention_dictionary`, which maps names like `"causal"` and `"ssm"` to their corresponding classes.

## 2. Create Your New Module

The first step is to create a new Python file in the appropriate subdirectory of `variations/` and implement your new module. For example, if you are creating a new attention mechanism, you would create a new file in `variations/attention_variations/`.

Your new module should be a `torch.nn.Module` and should accept the `GPTConfig` object in its constructor. This will give you access to all the hyperparameters of the model.

## 3. Register Your Variation

Once you have implemented your module, you need to register it in the corresponding variation dictionary.

1.  **Import your new module** in the `__init__.py` file of the relevant `variations` subdirectory.
2.  **Add your module to the dictionary.** In the same `__init__.py` file, add a new entry to the dictionary, with a unique name for your variation and the class you just created.

For example, to add a new attention variation called `"my_attention"`, you would edit `variations/attention_variations/__init__.py` to look something like this:

```python
from .my_attention_module import MyAttention
from .causal_self_attention import CausalSelfAttention
# ... other imports

attention_dictionary = {
    "causal": CausalSelfAttention,
    "my_attention": MyAttention,
    # ... other entries
}
```

## 4. Add a Command-Line Argument

To make your new variation accessible from the training script, you need to add a new command-line argument to `train_args.py`.

Find the relevant argument (e.g., `--attention_variant`) and add your new variation's name to the `choices` list.

```python
model_group.add_argument(
    "--attention_variant",
    type=str,
    default="causal",
    choices=["causal", "my_attention", ...], # Add your new variation here
    help="Which attention variant to use for the Transformer blocks."
)
```

## 5. Test Your Variation

Now you can test your new variation by running the training script with the new command-line argument.

```bash
python train.py --dataset=shakespeare_char --attention_variant=my_attention
```

Make sure to thoroughly test your new module to ensure it is working correctly and that it is compatible with the rest of the model.