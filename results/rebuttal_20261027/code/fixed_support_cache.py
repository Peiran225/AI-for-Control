"""Exact fixed-support Transformer evaluation with reusable support K/V.

This is the arithmetic implementation validated against
generate_offgrid_policy_switching_diagnostics.fixed_support_query_logits.
It does not change the attention set, weights, or output action transform.
"""
import hashlib,math
import torch
from torch.nn import functional as F
from train_paper_pmp_kkt import time_features

class CachedSupport:
    def __init__(self,model,support):
        self.model,self.layers=model.eval(),[]
        with torch.inference_mode():
            hidden=model.input(time_features(support))
            for layer in model.encoder.layers:
                attn=layer.self_attn
                assert attn.batch_first and attn.bias_k is None and attn.bias_v is None
                assert not attn.add_zero_attn and attn._qkv_same_embed_dim
                context=layer.norm1(hidden) if layer.norm_first else hidden
                width,heads=attn.embed_dim,attn.num_heads
                projected=F.linear(context,attn.in_proj_weight,attn.in_proj_bias)
                _,keys,values=projected.chunk(3,dim=-1)
                keys=keys.reshape(-1,heads,width//heads).transpose(0,1)
                values=values.reshape(-1,heads,width//heads).transpose(0,1)
                self.layers.append((layer,keys,values))
                hidden=layer(hidden.unsqueeze(0)).squeeze(0)

    def query(self,query,chunk=1024):
        result=[]
        with torch.inference_mode():
            for start in range(0,len(query),chunk):
                hidden=self.model.input(time_features(query[start:start+chunk]))
                for layer,keys,values in self.layers:
                    attn=layer.self_attn;width,heads=attn.embed_dim,attn.num_heads
                    before=layer.norm1(hidden) if layer.norm_first else hidden
                    q=F.linear(before,attn.in_proj_weight[:width],attn.in_proj_bias[:width])
                    q=q.reshape(-1,heads,width//heads).transpose(0,1)
                    scores=(q@keys.transpose(1,2))/math.sqrt(width//heads)
                    attention=(scores.softmax(dim=-1)@values).transpose(0,1).reshape(-1,width)
                    attention=attn.out_proj(attention)
                    if layer.norm_first:
                        hidden=hidden+layer.dropout1(attention);f=layer.norm2(hidden)
                        hidden=hidden+layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(f)))))
                    else:
                        hidden=layer.norm1(hidden+layer.dropout1(attention))
                        hidden=layer.norm2(hidden+layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(hidden))))))
                if self.model.encoder.norm is not None:hidden=self.model.encoder.norm(hidden)
                result.append(self.model.output(hidden).squeeze(-1))
        return torch.cat(result)

def model_fingerprint(model):
    h=hashlib.sha256()
    for key,tensor in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()
