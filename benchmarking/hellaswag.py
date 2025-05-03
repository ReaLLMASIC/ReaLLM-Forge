import torch
import numpy as np
import random
from torch.nn import functional as F
from datasets import load_dataset

def compute_logprob_for_sequence(model, idx_cond, next_tokens, temperature=0.8):
    """
    Compute log probability of next_tokens given idx_cond using the model.
    Returns the sum of the log probabilities for the given next_tokens.
    """
    logprob_sum = 0.0
    with torch.no_grad():
        for token in next_tokens:
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            # Compute probabilities
            probs = F.softmax(logits, dim=-1)
            # Probability of the specific token
            token_prob = probs[0, token].item()
            logprob_sum += np.log(token_prob + 1e-12)  # avoid log(0)
            # Append token
            idx_cond = torch.cat((idx_cond, token.unsqueeze(0).unsqueeze(0)), dim=1)
    return logprob_sum

def evaluate_hellaswag_few_shot(model, encode, device, few_shot_examples, start_text, eval_iters=250, temperature=0.8, seed=1337):
    """
    Perform few-shot evaluation on the HellaSwag dataset using a random subset for the few-shot prompt.
    - few_shot_examples: number of validation examples to use as few-shot demonstrations.
    - start_text: initial prompt to which we append few-shot examples.
    - seed: random seed for reproducibility of the few-shot sample.
    """
    # Load HellaSwag dataset
    ds = load_dataset("rowan/hellaswag", split='validation', trust_remote_code=True)

    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    model.to(device)

    # Randomly sample indices for the few-shot examples
    total_examples = len(ds)
    print("total_examples: ", total_examples)
    if few_shot_examples > total_examples:
        raise ValueError(f"few_shot_examples ({few_shot_examples}) exceeds the total number of examples in the dataset ({total_examples}).")

    few_shot_indices = np.random.choice(total_examples, size=few_shot_examples, replace=False)
    print("few_shot_indices: ", few_shot_indices)
    remaining_indices = list(set(range(total_examples)) - set(few_shot_indices))
    eval_indices = np.random.choice(remaining_indices, size=eval_iters, replace=False)
    print(f"Number of evaluation examples: {len(eval_indices)}")

    # Construct the few-shot prompt
    if few_shot_examples > 0:
        few_shot_prompt = []
        for idx in few_shot_indices:
            example = ds[int(idx)]
            correct_ending = example['endings'][int(example['label'])]
            # Format the few-shot example
            # example_str = f"Context: {example['ctx']}\nCorrect Ending: {correct_ending}\n---\n"
            example_str = f"{example['ctx']} {correct_ending}\n---\n"
            few_shot_prompt.append(example_str)
        few_shot_prompt = "".join(few_shot_prompt)
    else:
        few_shot_prompt = ""

    base_prompt = start_text + few_shot_prompt

    print("base_prompt: ", base_prompt)

    # Evaluate on the remaining validation examples
    correct = 0
    total = 0

    # Iterate over the evaluation dataset
    for idx in eval_indices:
        example = ds[int(idx)]
        endings = example['endings']
        label = example['label']

        # Construct the prompt for this instance
        # instance_prompt = base_prompt + f"Context: {example['ctx']}\n"
        instance_prompt = base_prompt + f"{example['ctx']}"
        instance_tokens = torch.tensor(encode(instance_prompt), dtype=torch.long, device=device).unsqueeze(0)

        # Compute log-probabilities for each ending
        scores = []
        for ending_idx, candidate in enumerate(endings):
            candidate_tokens = torch.tensor(encode(" " + candidate), dtype=torch.long, device=device)
            # Compute normalized log-prob
            lp = compute_logprob_for_sequence(model, instance_tokens.clone(), candidate_tokens, temperature=temperature)
            normalized_lp = lp / len(candidate_tokens)
            scores.append(normalized_lp)

        # Predicted = argmax of scores
        predicted = np.argmax(scores)
        print("scores: ", scores)
        print("predicted: ", predicted)
        print("label: ", label)
        if int(predicted) == int(label):
            correct += 1
        total += 1

    # Calculate and print accuracy
    accuracy = correct / total if total > 0 else 0.0
    print(f"Few-Shot Evaluation Accuracy on HellaSwag (N={few_shot_examples}): {accuracy:.4f}")