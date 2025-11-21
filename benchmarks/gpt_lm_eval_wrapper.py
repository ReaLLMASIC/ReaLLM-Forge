# gpt_lm_eval_wrapper.py

import os
import json
from datetime import datetime
import lm_eval
import torch
import torch.nn.functional as F
import lm_eval.api.model as model_api

class NanoGPTLM(model_api.LM):
    def __init__(self, model, tokenizer_encode, tokenizer_decode, eot_token_id, device, max_new_tokens, batch_size=1, temperature=1.0, top_k=None):
        super().__init__()
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self._eot_token_id = eot_token_id
        self.device = device
        self._batch_size = batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

    @property
    def eot_token_id(self):
        return self._eot_token_id

    @property
    def max_length(self):
        return self.model.config.block_size

    @property
    def batch_size(self):
        return self._batch_size

    def tok_encode(self, string):
        return self.tokenizer_encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer_decode(tokens)

    def _model_call(self, inps):
        logits, _ = self.model(inps)
        return logits

    def _model_generate(self, context, max_length, eos_token_id):
        idx = context.to(self.device)
        idx = self.model.generate(
            idx,
            max_new_tokens=max_length, 
            temperature=self.temperature, 
            top_k=self.top_k
        )
        return idx

    def loglikelihood(self, requests):
        """
        requests: list of Instance with arguments (context, continuation)
        returns: list of (loglikelihood, is_greedy)
        """
        res = []
        bs = self.batch_size
        B  = self.model.config.block_size

        for i in range(0, len(requests), bs):
            batch = requests[i : i+bs]

            # unpack and tokenize
            ctx_toks  = [self.tok_encode(inst.arguments[0]) for inst in batch]
            cont_toks = [self.tok_encode(inst.arguments[1]) for inst in batch]

            # LEFT-TRUNCATE each (ctx + cont) to ≤ B+1 tokens
            truncated = []
            for c, t in zip(ctx_toks, cont_toks):
                seq = c + t
                if len(seq) > B + 1:
                    seq = seq[-(B + 1):]
                    # re-split so that `t` remains the last len(t) tokens
                    t = seq[-len(t):]
                    c = seq[: -len(t)]
                truncated.append((c, t))

            # PAD all to the same length
            max_len = max(len(c) + len(t) for c, t in truncated)
            inputs  = []
            for c, t in truncated:
                seq = c + t
                seq += [self.eot_token_id] * (max_len - len(seq))
                inputs.append(seq)

            inp_tensor = torch.tensor(inputs, device=self.device)

            # run the model on all but the last token → [B×(L-1)×V]
            with torch.no_grad():
                logits, _ = self.model(inp_tensor[:, :-1])
                logp      = F.log_softmax(logits, dim=-1)

            # now score each continuation
            for j, (c, t) in enumerate(truncated):
                Lc, Lt = len(c), len(t)
                if Lt == 0:
                    # no continuation → zero log-likelihood, but greedy=TRUE by convention
                    res.append((0.0, True))
                    continue

                # grab exactly the continuation slice
                slice_logits = logp[j, Lc : Lc+Lt]       # shape [Lt, V]
                targets      = inp_tensor[j, Lc : Lc+Lt] # shape [Lt]

                # safety: if something mismatched, trim to the shortest
                if slice_logits.size(0) != targets.size(0):
                    m = min(slice_logits.size(0), targets.size(0))
                    slice_logits = slice_logits[:m]
                    targets      = targets[:m]

                # gather per-token log-probs and sum them
                lp = slice_logits.gather(1, targets.unsqueeze(-1)).squeeze(-1)
                ll = lp.sum().item()

                # greedy‐flag: true iff every argmax matches the target
                preds  = slice_logits.argmax(dim=-1)
                greedy = bool((preds == targets).all().item())

                res.append((ll, greedy))

        return res

    def loglikelihood_rolling(self, requests):
        """
        requests: list of Instance with arguments (text,)
        returns: list of loglikelihoods
        """
        res = []
        bs = self.batch_size
        B  = self.model.config.block_size

        for i in range(0, len(requests), bs):
            batch = requests[i : i+bs]
            texts = [inst.arguments[0] for inst in batch]

            # 1) tokenize and left-truncate to B tokens
            toks = []
            for txt in texts:
                t = self.tok_encode(txt)
                if len(t) > B:
                    t = t[-B:]
                toks.append(t)

            # 2) pad to a common length (≤ B)
            max_len = max(len(t) for t in toks)
            inputs  = [t + [self.eot_token_id]*(max_len - len(t)) for t in toks]
            inp_tensor = torch.tensor(inputs, device=self.device)

            # 3) run through model (all but last token)
            with torch.no_grad():
                logits, _ = self.model(inp_tensor[:, :-1])
                logp = F.log_softmax(logits, dim=-1)

            # 4) for each sequence, sum log‐probs of its tokens
            for j, t in enumerate(toks):
                L = len(t)
                if L < 2:
                    # if only 0 or 1 token, define loglikelihood = 0
                    res.append(0.0)
                    continue
                # slice logits corresponding to positions 0..L-2 predicting tokens 1..L-1
                slice_logits = logp[j, : L-1]   # shape [L-1, V]
                targets = inp_tensor[j, 1 : L]  # shape [L-1]
                lp = slice_logits.gather(1, targets.unsqueeze(-1)).squeeze(-1)
                ll = lp.sum().item()
                res.append(ll)

        return res

    def generate_until(self, requests):
        res = []
        for inst in requests:
            input_str, control_gen_params = inst.arguments
            input_tokens = torch.tensor([self.tok_encode(input_str)], device=self.device)
            # list of stopping strings
            stop_strings = control_gen_params.get("until", [])
            idx, generated_text = self.model.generate_with_stop(
                input_tokens,
                max_new_tokens=self.max_new_tokens,
                stop_strings=stop_strings,
                decode=self.tokenizer_decode,
                temperature=self.temperature,
                top_k=self.top_k
            )
            res.append(generated_text)
        return res
    
    @classmethod
    def create_model(cls, model, encode_fn, decode_fn, args):
        max_new = getattr(args, "max_benchmark_tokens", None)
        if max_new is None:
            max_new = getattr(args, "max_new_tokens", None)

        return cls(
            model=model,
            tokenizer_encode=encode_fn,
            tokenizer_decode=decode_fn,
            eot_token_id=model.config.vocab_size - 1, # |endoftext| token is the last token in GPT2
            device=args.device,
            max_new_tokens=max_new,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    
    def evaluate_and_save(self, tasks, batch_size, out_dir, timestamp=None, results_output=None):
        """
        tasks: list of lm-eval task names, e.g. ["arc_easy","hellaswag"]
        batch_size: how many samples per batch
        out_dir: base directory to write results into
        timestamp: optional timestamp string to use in the output filename
        results_output: if given, exact file path to write results; otherwise
                        generates a timestamped file under out_dir
        returns: dict of raw evaluation results
        """
        print(f"Running LM-Eval on tasks: {','.join(tasks)}")

        results = lm_eval.simple_evaluate(
            model=self,
            tasks=tasks,
            batch_size=batch_size,
        )

        # print just the summarized metrics
        print(results["results"])

        # decide where to save
        if results_output:
            save_path = results_output
        else:
            if not timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{timestamp}_lm_eval_results.json"
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, fname)

        print(f"Saving lm-eval results to {save_path}")
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
            f.write("\n")

        return results
