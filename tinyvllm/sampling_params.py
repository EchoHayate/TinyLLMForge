from dataclasses import dataclass

@dataclass
class SamplingParams:
    # 控制生成文本的多样性，设置较低则文本概率越集中，重复性越高
    # 设置较高则文本的概率越发散，重复概率越低，富有创造性
    temperature: float = 1.0  
    # 模型生成的最大 token长度， 1 token = 0.75个英文单词/1-2两个中文字符
    max_tokens: int = 64
    # 是否忽略句子的结束符，忽略会导致的句子无自然结尾
    ignore_eos: bool = False