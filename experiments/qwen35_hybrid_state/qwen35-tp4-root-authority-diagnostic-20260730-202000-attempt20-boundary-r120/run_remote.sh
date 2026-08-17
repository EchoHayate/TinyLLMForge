set -eu
env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-201355-attempt20-bf16-gate-r119/source PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=2 TINYVLLM_DIST_PORT=58661 MASTER_PORT=58662 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -c '
import json,sys,torch
sys.path.insert(0,'"'"'/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-201355-attempt20-bf16-gate-r119/source/tools'"'"')
import qwen35_tp4_real_root_logit_correctness_preflight as preflight
import qwen35_tp4_engine_official_reference_executor as official
configuration=json.load(open('"'"'/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/input/executor_configuration.json'"'"'))
configuration['"'"'model_manifest_path'"'"']='"'"'/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model_manifest.json'"'"'
configuration['"'"'dist_port'"'"']=58661
configuration['"'"'master_port'"'"']=58662
assert configuration.pop('"'"'world_size'"'"') == 4
configuration['"'"'gpu_indices'"'"']=tuple(configuration['"'"'gpu_indices'"'"'])
configuration=official.executor_module.ExecutorConfiguration(**configuration)
case=preflight._load_frozen_prompt_cases()[0]
prompt=tuple(case.token_ids)
if case.case_id!='"'"'p17'"'"' or len(prompt)!=17: raise RuntimeError('"'"'unexpected frozen p17 case'"'"')
backend=official.TransformersGreedyReferenceBackend(configuration,gpu_index=2)
captures={}; hooks=[]
def tensor(output): return output[0] if isinstance(output,tuple) else output
def capture(name,value):
 if name not in captures: captures[name]=tensor(value).squeeze(0).detach().float().cpu().clone()
def pre_layer(index):
 def hook(_module,args,kwargs):
  value=args[0] if args else kwargs['"'"'hidden_states'"'"']
  capture(f'"'"'layer{index}_input'"'"',value)
 return hook
def save_output(name):
 def hook(_module,_args,output): capture(name,output); return output
 return hook
try:
 model=backend._model(); text=model.model
 hooks.append(text.embed_tokens.register_forward_hook(save_output('"'"'embed_output'"'"')))
 for index,layer in enumerate(text.layers[:4]):
  hooks.append(layer.register_forward_pre_hook(pre_layer(index),with_kwargs=True))
  hooks.append(layer.input_layernorm.register_forward_hook(save_output(f'"'"'layer{index}_input_norm'"'"')))
  mixer=getattr(layer,'"'"'self_attn'"'"',None)
  if mixer is None: mixer=getattr(layer,'"'"'linear_attn'"'"',None)
  if mixer is None: mixer=getattr(layer,'"'"'linear_attention'"'"',None)
  if mixer is None: raise RuntimeError(f'"'"'layer {index} mixer module missing'"'"')
  hooks.append(mixer.register_forward_hook(save_output(f'"'"'layer{index}_mixer'"'"')))
  hooks.append(layer.post_attention_layernorm.register_forward_hook(save_output(f'"'"'layer{index}_post_attention_norm'"'"')))
  hooks.append(layer.mlp.register_forward_hook(save_output(f'"'"'layer{index}_mlp'"'"')))
  hooks.append(layer.register_forward_hook(save_output(f'"'"'layer{index}_output'"'"')))
 input_ids=torch.tensor([prompt],dtype=torch.int64,device=torch.device('"'"'cuda:0'"'"'))
 with torch.inference_mode(): model(input_ids=input_ids,use_cache=False,return_dict=True)
 captures['"'"'prompt_token_ids'"'"']=torch.tensor(prompt,dtype=torch.int64)
 torch.save(captures,'"'"'/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/output/official.pt'"'"')
finally:
 for hook in hooks: hook.remove()
 backend.close()
'
env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-201355-attempt20-bf16-gate-r119/source PYTHONDONTWRITEBYTECODE=1 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-201355-attempt20-bf16-gate-r119/source/tools/qwen35_tp4_correctness_resource_policy.py guard --policy controlled_shared --gpu-indices 2,4,5,6 --baseline /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/input/resource_baseline.json --baseline-sha256 e307deb1673fe040d958cd6f6527c04b37c4d07fb57e7eb6e675566f43bb39d4 --ssh-target sitian@10.232.195.203
(env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-201355-attempt20-bf16-gate-r119/source PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=2,4,5,6 TINYVLLM_DIST_PORT=58661 MASTER_PORT=58662 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/input/native_worker.py --rank 0 --rendezvous tcp://127.0.0.1:58661 --source /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-201355-attempt20-bf16-gate-r119/source --output /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/output >/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/output/rank0.stdout 2>/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/output/rank0.stderr) &
(env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-201355-attempt20-bf16-gate-r119/source PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=2,4,5,6 TINYVLLM_DIST_PORT=58661 MASTER_PORT=58662 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/input/native_worker.py --rank 1 --rendezvous tcp://127.0.0.1:58661 --source /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-201355-attempt20-bf16-gate-r119/source --output /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/output >/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/output/rank1.stdout 2>/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/output/rank1.stderr) &
(env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-201355-attempt20-bf16-gate-r119/source PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=2,4,5,6 TINYVLLM_DIST_PORT=58661 MASTER_PORT=58662 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/input/native_worker.py --rank 2 --rendezvous tcp://127.0.0.1:58661 --source /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-201355-attempt20-bf16-gate-r119/source --output /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/output >/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/output/rank2.stdout 2>/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/output/rank2.stderr) &
(env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-201355-attempt20-bf16-gate-r119/source PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=2,4,5,6 TINYVLLM_DIST_PORT=58661 MASTER_PORT=58662 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/input/native_worker.py --rank 3 --rendezvous tcp://127.0.0.1:58661 --source /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-201355-attempt20-bf16-gate-r119/source --output /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/output >/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/output/rank3.stdout 2>/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/output/rank3.stderr) &
wait
env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-201355-attempt20-bf16-gate-r119/source PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES= TINYVLLM_DIST_PORT=58661 MASTER_PORT=58662 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -c '
import json,torch
root='"'"'/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-202000-attempt20-boundary-r120/output'"'"'
official=torch.load(root+'"'"'/official.pt'"'"',map_location='"'"'cpu'"'"',weights_only=False)
native=[torch.load(root+f'"'"'/rank{rank}.pt'"'"',map_location='"'"'cpu'"'"',weights_only=False) for rank in range(4)]
components=['"'"'embed_output'"'"']
for index in range(4):
 components.extend([f'"'"'layer{index}_input'"'"',f'"'"'layer{index}_input_norm'"'"',f'"'"'layer{index}_mixer'"'"',f'"'"'layer{index}_post_attention_norm'"'"',f'"'"'layer{index}_mlp'"'"',f'"'"'layer{index}_output'"'"'])
rows=[]
def compare(component,rank,left,right):
 if left.shape!=right.shape: raise RuntimeError(f'"'"'shape mismatch {component} rank={rank} {left.shape} {right.shape}'"'"')
 diff=(left-right).abs(); nonzero=torch.nonzero(diff.reshape(-1),as_tuple=False)
 rows.append({'"'"'component'"'"':component,'"'"'rank'"'"':rank,'"'"'max_abs_diff'"'"':float(diff.max()),'"'"'mean_abs_diff'"'"':float(diff.mean()),'"'"'nonzero_count'"'"':int(torch.count_nonzero(diff)),'"'"'first_nonzero_index'"'"':None if nonzero.numel()==0 else int(nonzero[0])})
for rank,right in enumerate(native):
 for component in components: compare(component,rank,official[component],right[component])
result={'"'"'schema_version'"'"':'"'"'qwen35.tp4-root-authority-boundary-probe.v1'"'"','"'"'prompt_case_id'"'"':'"'"'p17'"'"','"'"'prompt_tokens'"'"':17,'"'"'rows'"'"':rows}
open(root+'"'"'/attention_probe.json'"'"','"'"'w'"'"').write(json.dumps(result,sort_keys=True,separators=('"'"','"'"','"'"':'"'"'))+'"'"'
'"'"')
print(json.dumps(result,sort_keys=True,separators=('"'"','"'"','"'"':'"'"')))
'
