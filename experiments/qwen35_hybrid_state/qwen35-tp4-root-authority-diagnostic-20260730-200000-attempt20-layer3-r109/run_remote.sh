set -eu
env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=2 TINYVLLM_DIST_PORT=58571 MASTER_PORT=58572 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -c '
import json,sys,torch
sys.path.insert(0,'"'"'/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source/tools'"'"')
import qwen35_tp4_real_root_logit_correctness_preflight as preflight
import qwen35_tp4_engine_official_reference_executor as official
configuration=json.load(open('"'"'/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/input/executor_configuration.json'"'"'))
configuration['"'"'model_manifest_path'"'"']='"'"'/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model_manifest.json'"'"'
configuration['"'"'dist_port'"'"']=58571
configuration['"'"'master_port'"'"']=58572
assert configuration.pop('"'"'world_size'"'"') == 4
configuration['"'"'gpu_indices'"'"']=tuple(configuration['"'"'gpu_indices'"'"'])
configuration=official.executor_module.ExecutorConfiguration(**configuration)
case=preflight._load_frozen_prompt_cases()[0]; prompt=tuple(case.token_ids)
if case.case_id!='"'"'p17'"'"' or len(prompt)!=17: raise RuntimeError('"'"'unexpected frozen p17 case'"'"')
backend=official.TransformersGreedyReferenceBackend(configuration,gpu_index=2)
captures={}; hooks=[]
def save(name):
 def hook(_module,_args,output):
  if name not in captures: captures[name]=output.squeeze(0).detach().float().cpu().clone()
  return output
 return hook
try:
 model=backend._model(); mixer=model.model.layers[0].linear_attn
 hooks.append(mixer.in_proj_qkv.register_forward_hook(save('"'"'in_proj_qkv'"'"')))
 hooks.append(mixer.in_proj_z.register_forward_hook(save('"'"'in_proj_z'"'"')))
 hooks.append(mixer.in_proj_b.register_forward_hook(save('"'"'in_proj_b'"'"')))
 hooks.append(mixer.in_proj_a.register_forward_hook(save('"'"'in_proj_a'"'"')))
 hooks.append(mixer.register_forward_hook(save('"'"'mixer_output'"'"')))
 input_ids=torch.tensor([prompt],dtype=torch.int64,device=torch.device('"'"'cuda:0'"'"'))
 with torch.inference_mode(): model(input_ids=input_ids,use_cache=False,return_dict=True)
 torch.save(captures,'"'"'/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/output/official.pt'"'"')
finally:
 for hook in hooks: hook.remove()
 backend.close()
'
env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source PYTHONDONTWRITEBYTECODE=1 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source/tools/qwen35_tp4_correctness_resource_policy.py guard --policy controlled_shared --gpu-indices 2,4,5,6 --baseline /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/input/resource_baseline.json --baseline-sha256 1f50a25d95d66cc831ff7889c2e94bf47461c466306ecdb6965dea0ee6ca74d1 --ssh-target sitian@10.232.195.203
(env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=2,4,5,6 TINYVLLM_DIST_PORT=58571 MASTER_PORT=58572 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/input/native_worker.py --rank 0 --rendezvous tcp://127.0.0.1:58571 --source /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source --output /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/output >/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/output/rank0.stdout 2>/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/output/rank0.stderr) &
(env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=2,4,5,6 TINYVLLM_DIST_PORT=58571 MASTER_PORT=58572 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/input/native_worker.py --rank 1 --rendezvous tcp://127.0.0.1:58571 --source /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source --output /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/output >/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/output/rank1.stdout 2>/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/output/rank1.stderr) &
(env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=2,4,5,6 TINYVLLM_DIST_PORT=58571 MASTER_PORT=58572 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/input/native_worker.py --rank 2 --rendezvous tcp://127.0.0.1:58571 --source /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source --output /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/output >/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/output/rank2.stdout 2>/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/output/rank2.stderr) &
(env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=2,4,5,6 TINYVLLM_DIST_PORT=58571 MASTER_PORT=58572 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/input/native_worker.py --rank 3 --rendezvous tcp://127.0.0.1:58571 --source /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source --output /data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/output >/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/output/rank3.stdout 2>/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/output/rank3.stderr) &
wait
env PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-root-logit-tests/qwen35-tp4-source-prep-20260730-180031-attempt20-current-r102/source PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES= TINYVLLM_DIST_PORT=58571 MASTER_PORT=58572 /data00/home/sitian/sitian-workspace01/tllm/env/bin/python -c '
import json,torch
root='"'"'/data00/home/sitian/sitian-workspace01/tllm/qwen35-tp4-layer-probes/qwen35-tp4-root-authority-diagnostic-20260730-200000-attempt20-layer3-r109/output'"'"'
o=torch.load(root+'"'"'/official.pt'"'"',map_location='"'"'cpu'"'"',weights_only=False)
ns=[torch.load(root+f'"'"'/rank{rank}.pt'"'"',map_location='"'"'cpu'"'"',weights_only=False) for rank in range(4)]
rows=[]
def compare(component,rank,left,right):
 if left.shape!=right.shape: raise RuntimeError(f'"'"'shape mismatch {component} rank={rank} {left.shape} {right.shape}'"'"')
 d=(left-right).abs(); nz=torch.nonzero(d.reshape(-1),as_tuple=False)
 rows.append({'"'"'component'"'"':component,'"'"'rank'"'"':rank,'"'"'max_abs_diff'"'"':float(d.max()),'"'"'mean_abs_diff'"'"':float(d.mean()),'"'"'nonzero_count'"'"':int(torch.count_nonzero(d)),'"'"'first_nonzero_index'"'"':None if nz.numel()==0 else int(nz[0])})
for rank,n in enumerate(ns):
 q,k,v=o['"'"'in_proj_qkv'"'"'].split((2048,2048,2048),dim=-1)
 compare('"'"'in_proj_qkv'"'"',rank,torch.cat([q[:,rank*512:(rank+1)*512],k[:,rank*512:(rank+1)*512],v[:,rank*512:(rank+1)*512]],dim=-1),n['"'"'in_proj_qkv'"'"'])
 compare('"'"'in_proj_z'"'"',rank,o['"'"'in_proj_z'"'"'][:,rank*512:(rank+1)*512],n['"'"'in_proj_z'"'"'])
 compare('"'"'in_proj_b'"'"',rank,o['"'"'in_proj_b'"'"'][:,rank*4:(rank+1)*4],n['"'"'in_proj_b'"'"'])
 compare('"'"'in_proj_a'"'"',rank,o['"'"'in_proj_a'"'"'][:,rank*4:(rank+1)*4],n['"'"'in_proj_a'"'"'])
 compare('"'"'mixer_output'"'"',rank,o['"'"'mixer_output'"'"'],n['"'"'mixer_output'"'"'])
result={'"'"'schema_version'"'"':'"'"'qwen35.tp4-root-authority-layer0-projection-probe.v1'"'"','"'"'prompt_case_id'"'"':'"'"'p17'"'"','"'"'prompt_tokens'"'"':17,'"'"'rows'"'"':rows}
open(root+'"'"'/attention_probe.json'"'"','"'"'w'"'"').write(json.dumps(result,sort_keys=True,separators=('"'"','"'"','"'"':'"'"'))+'"'"'
'"'"')
print(json.dumps(result,sort_keys=True,separators=('"'"','"'"','"'"':'"'"')))
'
