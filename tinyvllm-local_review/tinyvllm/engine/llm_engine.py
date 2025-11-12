from tinyvllm.sampling_params import SamplingParams
from tinyvllm.config import Config
from tinyvllm.engine.model_runner import ModelRunner
from tinyvllm.engine.scheduler import Scheduler
from tinyvllm.engine.sequence import Sequence

from dataclasses import fields

from time import perf_counter
import atexit
from tqdm.auto import tqdm
import torch.multiprocessing as mp
from transformers import AutoTokenizer

class LLMEngine:
    
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}    #过滤掉和config无关的参数
        config  = Config(model, **config_kwargs)       
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")                       # 生成全新解释器，继承基本资源，全局变量，打开的文件，线程不会被继承
        for i in range(1, config.tensor_parallel_size):     # 生成所有的子进程
            event = ctx.Event()                             #进程间同步的“信号量”，用于进程间通信
            process = ctx.Process(target=ModelRunner, args = {config, i, event}) #创建子进程对象 modelrunner是子进程要执行的目标函数
            process.start()
            self.ps.append(process)
            self.events.append(event)

        self.model_runner = ModelRunner(config, 0, self.events)     # 生成主进程
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast = True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)
        
    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner                           # 显式释放
        for p in self.ps:
            p.join()

    def add_request(
        self, 
        prompt: str | list[int], 
        sampling_params: SamplingParams
    ):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)           #直接加到waiting

    def step(self):     #decode阶段：每次step生成新的token加到seq后面
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)     
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]       #output包含seq_id和已经生成的token列表
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -(len(seqs))      #因为decode每个sequence只生成一个token 所以seqs的数量就是token的数量        
        return outputs, num_tokens      #计算的是每个step的单次增量

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self, 
        prompts: list[str] | list[list[int]],               #输入提示：可以是字符串列表（未分词）也可以是token id列表（已分词）
        sampling_params: SamplingParams | list[SamplingParams], 
        use_tqdm: bool = True, 
    ) -> list[int]:
        if use_tqdm: 
            pbar = tqdm(total = len(prompts), desc = "Generating", dynamic_ncols = True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)    #保证每个prompt都有一组sampling_params
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():           #根据waiting和running队列是否为空判断
            t = perf_counter()                  #纳秒级别的高精度时间（自计算机启动经过的时间）
            output, num_tokens = self.step()
            if use_tqdm:
                # prefill
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                # decode
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)  #为了区分decode和prefill 规定decode阶段的num_tokens都是-1 （decode每个step阶段都是生成1个token）
                pbar.set_postfix({
                    "prefill": f"{int(prefill_throughput)} tok/s",     #一次step的吞吐
                    "Decode": f"{int(decode_throughput)} tok/s"
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
    