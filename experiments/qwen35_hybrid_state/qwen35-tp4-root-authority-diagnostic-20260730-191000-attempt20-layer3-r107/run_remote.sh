set -eu
env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=2 TINYVLLM_DIST_PORT=58551 MASTER_PORT=58552 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -c '
import json,sys,torch
sys.path.insert(0,'"'"'/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source/tools'"'"')
import qwen35_tp4_real_root_logit_correctness_preflight as preflight
import qwen35_tp4_engine_official_reference_executor as official
configuration=json.load(open('"'"'/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/input/executor_configuration.json'"'"'))
configuration['"'"'model_manifest_path'"'"']='"'"'/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model_manifest.json'"'"'
configuration['"'"'dist_port'"'"']=58551
configuration['"'"'master_port'"'"']=58552
assert configuration.pop('"'"'world_size'"'"') == 4
configuration['"'"'gpu_indices'"'"']=tuple(configuration['"'"'gpu_indices'"'"'])
configuration=official.executor_module.ExecutorConfiguration(**configuration)
case=preflight._load_frozen_prompt_cases()[0]
prompt=tuple(case.token_ids)
if len(prompt)!=17: raise RuntimeError(f'"'"'unexpected p17 length: {len(prompt)}'"'"')
backend=official.TransformersGreedyReferenceBackend(configuration,gpu_index=2)
captures={}; hooks=[]
def tensor(output): return output[0] if isinstance(output,tuple) else output
def save_output(name):
 def hook(_module,_args,output):
  if name not in captures: captures[name]=tensor(output).squeeze(0).detach().float().cpu().clone()
  return output
 return hook
def pre_hidden(_module,args,kwargs):
 if '"'"'attention_input'"'"' not in captures:
  value=args[0] if args else kwargs['"'"'hidden_states'"'"']
  captures['"'"'attention_input'"'"']=value.squeeze(0).detach().float().cpu().clone()
def pre_input(_module,args):
 if '"'"'output_projection_input'"'"' not in captures:
  captures['"'"'output_projection_input'"'"']=args[0].squeeze(0).detach().float().cpu().clone()
try:
 model=backend._model(); attention=model.model.layers[3].self_attn
 official_module=sys.modules[attention.__class__.__module__]
 original_rotary=official_module.apply_rotary_pos_emb
 original_eager=official_module.eager_attention_forward
 def rotary(query,key,cos,sin,*args,**kwargs):
  output=original_rotary(query,key,cos,sin,*args,**kwargs)
  if '"'"'rotated_query'"'"' not in captures:
   captures['"'"'rotated_query'"'"']=output[0].transpose(1,2).squeeze(0).detach().float().cpu().clone()
   captures['"'"'rotated_key'"'"']=output[1].transpose(1,2).squeeze(0).detach().float().cpu().clone()
  return output
 def eager(module,query,key,value,attention_mask,scaling,dropout=0.0,**kwargs):
  first='"'"'attention_query'"'"' not in captures
  if first:
   captures['"'"'attention_query'"'"']=query.transpose(1,2).squeeze(0).detach().float().cpu().clone()
   captures['"'"'attention_key'"'"']=key.transpose(1,2).squeeze(0).detach().float().cpu().clone()
   captures['"'"'attention_value'"'"']=value.transpose(1,2).squeeze(0).detach().float().cpu().clone()
  output=original_eager(module,query,key,value,attention_mask,scaling,dropout=dropout,**kwargs)
  if first: captures['"'"'attention_output'"'"']=output[0].squeeze(0).detach().float().cpu().clone()
  return output
 official_module.apply_rotary_pos_emb=rotary
 official_module.eager_attention_forward=eager
 hooks.append(attention.register_forward_pre_hook(pre_hidden,with_kwargs=True))
 hooks.append(attention.q_proj.register_forward_hook(save_output('"'"'q_projection'"'"')))
 hooks.append(attention.k_proj.register_forward_hook(save_output('"'"'k_projection'"'"')))
 hooks.append(attention.v_proj.register_forward_hook(save_output('"'"'v_projection'"'"')))
 hooks.append(attention.q_norm.register_forward_hook(save_output('"'"'q_norm'"'"')))
 hooks.append(attention.k_norm.register_forward_hook(save_output('"'"'k_norm'"'"')))
 hooks.append(attention.o_proj.register_forward_pre_hook(pre_input))
 hooks.append(attention.register_forward_hook(save_output('"'"'mixer_output'"'"')))
 input_ids=torch.tensor([prompt],dtype=torch.int64,device=torch.device('"'"'cuda:0'"'"'))
 with torch.inference_mode(): model(input_ids=input_ids,use_cache=False,return_dict=True)
 paired=captures['"'"'q_projection'"'"'].view(17,8,512)
 captures['"'"'query_projection'"'"']=paired[...,:256].clone()
 captures['"'"'query_gate'"'"']=paired[...,256:].reshape(17,-1).clone()
 captures['"'"'gated_output'"'"']=captures['"'"'output_projection_input'"'"'].clone()
 captures['"'"'prompt_token_ids'"'"']=torch.tensor(prompt,dtype=torch.int64)
 torch.save(captures,'"'"'/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/output/official.pt'"'"')
finally:
 for hook in hooks: hook.remove()
 if '"'"'official_module'"'"' in locals():
  official_module.apply_rotary_pos_emb=original_rotary
  official_module.eager_attention_forward=original_eager
 backend.close()
'
env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source PYTHONDONTWRITEBYTECODE=1 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source/tools/qwen35_tp4_correctness_resource_policy.py guard --policy controlled_shared --gpu-indices 2,4,5,6 --baseline /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/input/resource_baseline.json --baseline-sha256 0ad30011eff74d59a0743bb31820164730bb527aed64e2f832af4e72ecd1e436 --ssh-target sitian@10.232.195.203
(env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=2,4,5,6 TINYVLLM_DIST_PORT=58551 MASTER_PORT=58552 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/input/native_worker.py --rank 0 --rendezvous tcp://127.0.0.1:58551 --source /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source --output /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/output >/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/output/rank0.stdout 2>/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/output/rank0.stderr) &
(env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=2,4,5,6 TINYVLLM_DIST_PORT=58551 MASTER_PORT=58552 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/input/native_worker.py --rank 1 --rendezvous tcp://127.0.0.1:58551 --source /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source --output /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/output >/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/output/rank1.stdout 2>/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/output/rank1.stderr) &
(env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=2,4,5,6 TINYVLLM_DIST_PORT=58551 MASTER_PORT=58552 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/input/native_worker.py --rank 2 --rendezvous tcp://127.0.0.1:58551 --source /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source --output /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/output >/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/output/rank2.stdout 2>/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/output/rank2.stderr) &
(env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=2,4,5,6 TINYVLLM_DIST_PORT=58551 MASTER_PORT=58552 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/input/native_worker.py --rank 3 --rendezvous tcp://127.0.0.1:58551 --source /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source --output /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/output >/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/output/rank3.stdout 2>/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/output/rank3.stderr) &
wait
env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES= TINYVLLM_DIST_PORT=58551 MASTER_PORT=58552 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -c '
import json,torch
root='"'"'/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-191000-attempt20-layer3-r107/output'"'"'
official=torch.load(root+'"'"'/official.pt'"'"',map_location='"'"'cpu'"'"',weights_only=False)
native=[torch.load(root+f'"'"'/rank{rank}.pt'"'"',map_location='"'"'cpu'"'"',weights_only=False) for rank in range(4)]
rows=[]
def compare(component,rank,left,right):
 if left.shape!=right.shape: raise RuntimeError(f'"'"'shape mismatch {component} rank={rank} {left.shape} {right.shape}'"'"')
 diff=(left-right).abs(); nonzero=torch.nonzero(diff.reshape(-1),as_tuple=False)
 rows.append({'"'"'component'"'"':component,'"'"'rank'"'"':rank,'"'"'max_abs_diff'"'"':float(diff.max()),'"'"'mean_abs_diff'"'"':float(diff.mean()),'"'"'nonzero_count'"'"':int(torch.count_nonzero(diff)),'"'"'first_nonzero_index'"'"':None if nonzero.numel()==0 else int(nonzero[0])})
for rank,right in enumerate(native):
 q_start=rank*2; kv_start=rank//2
 compare('"'"'attention_input'"'"',rank,official['"'"'attention_input'"'"'],right['"'"'attention_input'"'"'])
 compare('"'"'q_projection'"'"',rank,official['"'"'q_projection'"'"'].view(17,8,512)[:,q_start:q_start+2].reshape(17,-1),right['"'"'q_projection'"'"'])
 compare('"'"'query_projection'"'"',rank,official['"'"'query_projection'"'"'][:,q_start:q_start+2],right['"'"'query_projection'"'"'])
 compare('"'"'k_projection'"'"',rank,official['"'"'k_projection'"'"'].view(17,2,256)[:,kv_start:kv_start+1].reshape(17,-1),right['"'"'k_projection'"'"'])
 compare('"'"'v_projection'"'"',rank,official['"'"'v_projection'"'"'].view(17,2,256)[:,kv_start:kv_start+1].reshape(17,-1),right['"'"'v_projection'"'"'])
 compare('"'"'q_norm'"'"',rank,official['"'"'q_norm'"'"'][:,q_start:q_start+2],right['"'"'q_norm'"'"'])
 compare('"'"'k_norm'"'"',rank,official['"'"'k_norm'"'"'][:,kv_start:kv_start+1],right['"'"'k_norm'"'"'])
 compare('"'"'rotated_query'"'"',rank,official['"'"'rotated_query'"'"'][:,q_start:q_start+2].reshape(17,-1),right['"'"'rotated_query'"'"'])
 compare('"'"'rotated_key'"'"',rank,official['"'"'rotated_key'"'"'][:,kv_start:kv_start+1].reshape(17,-1),right['"'"'rotated_key'"'"'])
 compare('"'"'attention_query'"'"',rank,official['"'"'attention_query'"'"'][:,q_start:q_start+2].reshape(17,-1),right['"'"'attention_query'"'"'])
 compare('"'"'attention_key'"'"',rank,official['"'"'attention_key'"'"'][:,kv_start:kv_start+1].reshape(17,-1),right['"'"'attention_key'"'"'])
 compare('"'"'attention_value'"'"',rank,official['"'"'attention_value'"'"'][:,kv_start:kv_start+1].reshape(17,-1),right['"'"'attention_value'"'"'])
 compare('"'"'attention_output'"'"',rank,official['"'"'attention_output'"'"'][:,q_start:q_start+2].reshape(17,-1),right['"'"'attention_output'"'"'])
 compare('"'"'query_gate'"'"',rank,official['"'"'query_gate'"'"'].view(17,8,256)[:,q_start:q_start+2].reshape(17,-1),right['"'"'query_gate'"'"'])
result={'"'"'schema_version'"'"':'"'"'qwen35.tp4-root-authority-layer3-probe.v1'"'"','"'"'prompt_case_id'"'"':'"'"'p17'"'"','"'"'prompt_tokens'"'"':17,'"'"'rows'"'"':rows}
open(root+'"'"'/attention_probe.json'"'"','"'"'w'"'"').write(json.dumps(result,sort_keys=True,separators=('"'"','"'"','"'"':'"'"'))+'"'"'\n'"'"')
print(json.dumps(result,sort_keys=True,separators=('"'"','"'"','"'"':'"'"')))
'
