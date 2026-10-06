"""Optional whole-summary contextual-token matching using an HF encoder.

Long summaries are independently encoded in blocks; every token is covered.
This is BERTScore-style matching, NOT the canonical BERTScore implementation.
No claim of factuality is made. Run separately from generation timing/RAM tests.
"""
from pathlib import Path
import json
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModel
from huggingface_hub import HfApi
from ..storage import append_json
from .report import completed_rows
from .config import file_digest,digest


@torch.no_grad()
def semantic(files,output,model_id="sentence-transformers/all-MiniLM-L6-v2",revision="main",device="cpu",block_tokens=192):
    output=Path(output)
    if output.exists() or output.with_suffix('.meta.json').exists():raise FileExistsError(output)
    if block_tokens<1:raise ValueError('block_tokens must be positive')
    rows,_=completed_rows(files)
    sha=HfApi().model_info(model_id,revision=revision).sha
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=sha)
    model=AutoModel.from_pretrained(model_id,revision=sha).to(device).eval()
    limit=min(getattr(model.config,"max_position_embeddings",512),tokenizer.model_max_length)
    if block_tokens+tokenizer.num_special_tokens_to_add()>limit:raise ValueError("Semantic block exceeds encoder context")
    def embeddings(text):
        ids=tokenizer.encode(text,add_special_tokens=False,truncation=False,verbose=False);chunks=[]
        for i in range(0,len(ids),block_tokens):
            batch=tokenizer.prepare_for_model(ids[i:i+block_tokens],return_tensors="pt",
                prepend_batch_axis=True,return_attention_mask=True,verbose=False)
            special=torch.tensor([tokenizer.get_special_tokens_mask(batch['input_ids'][0].tolist(),
                already_has_special_tokens=True)],dtype=torch.bool,device=device)
            batch={k:v.to(device) for k,v in batch.items()}
            h=model(**batch).last_hidden_state
            chunks.append(F.normalize(h[~special].float(),dim=-1))
        return torch.cat(chunks) if chunks else None
    output.parent.mkdir(parents=True,exist_ok=True)
    meta={'status':'in_progress','encoder':model_id,'encoder_revision':sha,'block_tokens':block_tokens,
          'inputs':{str(f):file_digest(f) for f in files}}
    output.with_suffix('.meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    output.touch(exist_ok=False)
    for row in rows:
        if row["status"]!="ok":continue
        p,r=embeddings(row["prediction"]),embeddings(row["reference"])
        if p is None or r is None:precision=recall=f1=0.0
        else:
            similarity=p@r.T
            precision=float(similarity.max(1).values.mean());recall=float(similarity.max(0).values.mean())
            f1=2*precision*recall/(precision+recall) if precision+recall>0 else 0.0
        append_json(output,{"run_id":row["run_id"],"source_sha256":row["source_sha256"],
            "reference_sha256":row['reference_sha256'],"prediction_sha256":digest(row['prediction']),
            "prediction_encoder_tokens":0 if p is None else len(p),
            "reference_encoder_tokens":0 if r is None else len(r),
            "semantic_precision":precision,"semantic_recall":recall,"semantic_f1":f1,
            "encoder":model_id,"encoder_revision":sha,"block_tokens":block_tokens,
            "recipe":"whole-summary chunked contextual token matching; no IDF or baseline rescaling",
            "canonical_bertscore":False,"factuality_metric":False})
    meta.update(status='complete',sha256=file_digest(output))
    output.with_suffix('.meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    return str(output)
