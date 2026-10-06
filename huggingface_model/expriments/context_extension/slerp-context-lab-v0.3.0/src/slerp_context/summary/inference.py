"""Matched greedy decoding with explicit source/compression/answer costs."""
from dataclasses import dataclass,asdict
from contextlib import nullcontext
import time
import torch
from ..rmt import RMTLM,cache_bytes
from .data import prompt_ids,prompt_parts,encode


BASELINES={"native_full","head","tail","head_tail","rolling","map_reduce"}
LEARNED={"local","nlerp","slerp","delta","rmt","ema","fixed"}


def sync(device):
    if device.type=="cuda":torch.cuda.synchronize(device)


def amp(device):
    return torch.autocast("cuda",dtype=torch.bfloat16) if device.type=="cuda" else nullcontext()


@dataclass
class Generated:
    ids:list
    input_tokens:int
    generated_tokens:int
    first_token_at:float
    elapsed_s:float
    cache_bytes:int
    limit_hit:bool


class Engine:
    def __init__(self,model,tokenizer,study):
        self.model,self.tokenizer,self.study=model,tokenizer,study
        self.device=next(model.parameters()).device
        self.native=model.native_context
        cfg=model.original.generation_config
        eos=cfg.eos_token_id if cfg.eos_token_id is not None else tokenizer.eos_token_id
        self.eos=set(eos if isinstance(eos,(list,tuple)) else [eos])
        self.model.eval()

    @torch.no_grad()
    def generate(self,prompt,budget,recurrent=False):
        if not prompt or budget<1:raise ValueError("Nonempty prompt and positive output budget required")
        if not recurrent and len(prompt)+budget>self.native:
            raise ValueError("Native context would be exceeded")
        x=torch.tensor([prompt],device=self.device)
        sync(self.device);start=time.perf_counter()
        with amp(self.device):
            if recurrent:
                if isinstance(self.model,RMTLM):logits,state=self.model.prefill(x)
                else:logits,state=self.model.consume(x,all_logits=False)
                cache=None
            else:
                out=self.model.backbone.model(input_ids=x,use_cache=True)
                cache=out.past_key_values;state=None
                logits=self.model.backbone.get_output_embeddings()(out.last_hidden_state[:,-1:])
            generated=[];first=None
            for i in range(budget):
                token=logits[:,-1].float().argmax(-1,keepdim=True)
                value=int(token.item())
                if first is None:
                    sync(self.device);first=time.perf_counter()
                generated.append(value)
                if value in self.eos or i+1==budget:break
                if recurrent:
                    if isinstance(self.model,RMTLM):logits,state=self.model.step(token,state)
                    else:logits,state=self.model.consume(token,state,all_logits=False)
                else:
                    out=self.model.backbone.model(input_ids=token,past_key_values=cache,use_cache=True)
                    cache=out.past_key_values
                    logits=self.model.backbone.get_output_embeddings()(out.last_hidden_state[:,-1:])
        sync(self.device);elapsed=time.perf_counter()-start
        size=state.nbytes() if recurrent else cache_bytes(cache)
        return Generated(generated,len(prompt),len(generated),first,elapsed,size,
                         len(generated)==budget and generated[-1] not in self.eos)

    def text(self,ids):return self.tokenizer.decode(ids,skip_special_tokens=True).strip()


def select_source(ids,budget,method):
    if budget<1:raise ValueError("Prompt/output reservations leave no source room")
    if len(ids)<=budget:return list(ids)
    if method=="head":return list(ids[:budget])
    if method=="tail":return list(ids[-budget:])
    if method=="head_tail":
        head=(budget+1)//2
        return list(ids[:head])+list(ids[-(budget-head):]) if budget>1 else list(ids[:1])
    raise ValueError("Unknown truncation")


def run_document(engine,row,method):
    study,tokenizer=engine.study,engine.tokenizer
    source=row["source_ids"];calls=[]
    sync(engine.device);start=time.perf_counter()
    def call(ids,budget,instruction=None,recurrent=False):
        prompt=prompt_ids(tokenizer,study,ids,instruction)
        if not recurrent and method!="native_full" and len(prompt)+budget>study.base().window:
            raise ValueError("A baseline call exceeded its declared working budget")
        result=engine.generate(prompt,budget,recurrent)
        calls.append(result)
        return result
    final_budget=study.max_new_tokens
    prefix,suffix=prompt_parts(tokenizer,study)
    source_budget=study.base().window-len(prefix)-len(suffix)-final_budget
    if method=="native_full":
        if len(source)+len(prefix)+len(suffix)+final_budget>engine.native:
            return {"status":"skipped_native_limit","prediction":None,"source_tokens_seen":0,"calls":[]}
        final=call(source,final_budget);seen=len(source)
    elif method in {"head","tail","head_tail"}:
        selected=select_source(source,source_budget,method)
        final=call(selected,final_budget);seen=len(selected)
    elif method in LEARNED:
        if method!=engine.model.cfg.method:raise ValueError("Requested memory method differs from loaded weights")
        final=call(source,final_budget,recurrent=True);seen=len(source)
    elif method=="rolling":
        # The entire prior summary is preserved. Source chunk size reserves the
        # maximum intermediate output so every call provably fits the budget.
        instruction=("Update the running executive summary using the new passage. "
                     "Preserve earlier major findings, qualifications and conclusions. "
                     "Do not invent details. Return only the updated summary.")
        a,b=prompt_parts(tokenizer,study,instruction)
        left=encode(tokenizer,"Running summary:\n");mid=encode(tokenizer,"\nNew passage:\n")
        capacity=study.base().window-len(a)-len(b)-len(left)-len(mid)-2*study.intermediate_tokens
        if capacity<32:raise ValueError("Working window too small for rolling summaries")
        memory=[]
        for pos in range(0,len(source),capacity):
            g=call(left+memory+mid+source[pos:pos+capacity],study.intermediate_tokens,instruction)
            memory=encode(tokenizer,engine.text(g.ids))[:study.intermediate_tokens]
        final=call(memory,final_budget);seen=len(source)
    elif method=="map_reduce":
        instruction=("Summarize this passage, preserving its main findings, explanations, "
                     "qualifications and conclusions. Return only the summary.")
        a,b=prompt_parts(tokenizer,study,instruction)
        capacity=study.base().window-len(a)-len(b)-study.intermediate_tokens
        if capacity<32 or source_budget<study.intermediate_tokens:
            raise ValueError("Working window too small for hierarchical summaries")
        pieces=[]
        for pos in range(0,len(source),capacity):
            g=call(source[pos:pos+capacity],study.intermediate_tokens,instruction)
            pieces.append(encode(tokenizer,engine.text(g.ids)))
        separator=encode(tokenizer,"\n\n")
        def flatten(parts):
            joined=[]
            for part in parts:
                if joined:joined+=separator
                joined+=part
            return joined
        for _ in range(32):
            joined=flatten(pieces)
            if len(joined)<=source_budget:break
            groups=[];group=[]
            for piece in pieces:
                candidate=flatten(group+[piece])
                if len(candidate)>capacity and group:groups.append(flatten(group));group=[]
                group.append(piece)
            if group:groups.append(flatten(group))
            updated=[]
            for group in groups:
                g=call(group,study.intermediate_tokens,instruction)
                updated.append(encode(tokenizer,engine.text(g.ids)))
            if len(flatten(updated))>=len(joined):
                raise RuntimeError("Hierarchical summaries failed to shrink; lower intermediate_tokens")
            pieces=updated
        else:raise RuntimeError("Too many hierarchy levels")
        final=call(flatten(pieces),final_budget);seen=len(source)
    else:raise ValueError("Unknown summarization method")
    sync(engine.device);total=time.perf_counter()-start
    details=[asdict(c) for c in calls]
    for c in details:c.pop("ids")
    return {"status":"ok","prediction":engine.text(final.ids),"source_tokens_seen":seen,
        "end_to_end_s":total,"final_first_token_s":final.first_token_at-start,
        "all_generation_tokens":sum(c.generated_tokens for c in calls),
        "final_generation_tokens":final.generated_tokens,"all_call_input_tokens":sum(c.input_tokens for c in calls),
        "max_call_cache_bytes":max(c.cache_bytes for c in calls),"final_cache_bytes":final.cache_bytes,
        "generation_limit_hit":final.limit_hit,"intermediate_limit_hits":sum(c.limit_hit for c in calls[:-1]),
        "num_calls":len(calls),"calls":details}
