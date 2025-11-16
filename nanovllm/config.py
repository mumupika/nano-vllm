import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str                                  # The model path.
    max_num_batched_tokens: int = 16384         # Maximum number of tokens to be processed in a single iteration.
    max_num_seqs: int = 512                     # Maximum number of sequences to be processed in a single iteration.
    max_model_len: int = 4096                   # Model context length (prompt and output).
    gpu_memory_utilization: float = 0.9         # The fraction of GPU memory to be used for the model executor.
    tensor_parallel_size: int = 1               # tensor parallel dim. See megatron's essay.
    enforce_eager: bool = False                 # Whether use eager mode(one line after one line) or graph mode of pytorch.
    hf_config: AutoConfig | None = None
    eos: int = -1                               # end of sequence token.
    kvcache_block_size: int = 256               # The size of each kvcache.
    num_kvcache_blocks: int = -1                # The numbers of kvcache blocks.

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
