import os
from dataclasses import dataclass
from transformers import AutoConfig
import torch
from ssd.paths import DEFAULT_TARGET, DEFAULT_DRAFT

@dataclass
class Config:
    model: str = DEFAULT_TARGET
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 1 
    max_model_len: int = 4096 
    gpu_memory_utilization: float = 0.7
    enable_prefix_cache: bool = True
    draft_gpu_memory_utilization: float | None = None  # None = upstream default (0.75 sync / 0.8 async)
    early_publish_tokens: int = 0  # 0: complete-cache SSD; positive: uniform minimum ready depth
    publication_policy: str = 'uniform'  # uniform (v1), bounded, adaptive, or overlap
    publication_max_inflight: int = 2  # unfinished draft steps, bounded/adaptive only
    drain_draft_on_finish: bool = False
    greedy_transport: bool = False  # versioned IDs-only protocol; legacy stays default
    handoff_service: str = 'inline'  # inline or independent CPU service thread
    handoff_bypass_ready: bool = False
    handoff_trace: bool = False  # intrusive diagnostic; never headline timing
    target_numerics: str = 'batched'  # 'ar_compatible' opts into AR-shaped target verification.
    num_gpus: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # spec config args
    draft_hf_config: AutoConfig | None = None
    speculate: bool = False 
    draft: str = DEFAULT_DRAFT
    speculate_k: int = 1
    draft_async: bool = False
    
    # async spec only
    async_fan_out: int = 3
    fan_out_list: list[int] | None = None
    fan_out_list_miss: list[int] | None = None
    sampler_x: float | None = None 
    jit_speculate: bool = False 

    # eagle3
    use_eagle: bool = False 
    eagle_layers: list[int] | None = None   
    d_model_target: int | None = None
    tokenizer_path: str | None = None

    # Debugging
    verbose: bool = False 
    debug_mode: bool = False 
    max_steps: int | None = None

    @property
    def max_blocks(self): 
        return (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size

    def __post_init__(self):
        model = self.model 
        if self.handoff_service not in ('inline', 'threaded'):
            raise ValueError('Unknown handoff service')
        if self.greedy_transport:
            if (not self.speculate or not self.draft_async or self.max_num_seqs != 1 or
                    self.use_eagle or not self.jit_speculate or self.sampler_x is not None):
                raise ValueError('IDs-only transport requires plain batch-one async greedy SSD with JIT')
            self.drain_draft_on_finish = True
        elif self.handoff_service != 'inline' or self.handoff_bypass_ready or self.handoff_trace:
            raise ValueError('New handoff options require IDs-only transport')
        if self.publication_policy not in ('uniform', 'bounded', 'adaptive', 'overlap'):
            raise ValueError('Unknown publication policy')
        if not 1 <= self.publication_max_inflight <= 15:
            raise ValueError('Publication queue bound must be between 1 and 15')
        if self.publication_policy != 'uniform' and not self.early_publish_tokens:
            raise ValueError('Publication policy requires early publication enabled')
        if self.target_numerics not in ('batched', 'ar_compatible', 'shared_fixed16'):
            raise ValueError('Unknown target numerical backend')
        if self.target_numerics == 'shared_fixed16':
            if self.max_num_seqs != 1 or self.num_gpus - int(self.draft_async) != 1 or self.use_eagle:
                raise ValueError('Shared fixed16 requires batch one, one target GPU, and plain Qwen3')
            if not 1 <= self.speculate_k <= 15:
                raise ValueError('Shared fixed16 supports at most 15 speculative tokens')
        if self.drain_draft_on_finish and (not self.draft_async or self.max_num_seqs != 1 or self.use_eagle):
            raise ValueError("Draft drain accounting requires plain async batch-one SSD")
        if self.early_publish_tokens:
            if (not self.speculate or not self.draft_async or self.use_eagle or
                    self.max_num_seqs != 1 or not self.jit_speculate or
                    not 1 <= self.early_publish_tokens <= self.speculate_k):
                raise ValueError('Early publication requires async batch-one SSD, JIT misses, and 1 <= P <= K')
            self.drain_draft_on_finish = True
        assert os.path.isdir(model)

        assert 1 <= self.num_gpus <= 8 # this codebase only works on one node 
        self.hf_config = AutoConfig.from_pretrained(model)
        if self.target_numerics == 'ar_compatible' and self.speculate:
            from ssd.engine.ar_compatible import validate_ar_compatible
            validate_ar_compatible(self, self.hf_config.model_type,
                                   self.num_gpus - 1 if self.draft_async else self.num_gpus)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings) 
        if self.speculate: 
            draft = self.draft
            self.draft_hf_config = AutoConfig.from_pretrained(draft)
            self.max_model_len = min(
                self.max_model_len, self.draft_hf_config.max_position_embeddings)
            if self.draft_async:
                if self.fan_out_list is None: 
                    self.fan_out_list = [self.async_fan_out] * (self.speculate_k + 1)
                    self.MQ_LEN = sum(self.fan_out_list)
                if self.fan_out_list_miss is None:
                    self.fan_out_list_miss = self.fan_out_list 
                assert sum(self.fan_out_list_miss) == sum(self.fan_out_list), "ERROR in Config: fan_out_list_miss must be the same as fan_out_list"
                
        if self.use_eagle:
            if self.eagle_layers is None:
                L = self.hf_config.num_hidden_layers
                # self.eagle_layers = [3, L//2, L-3]
                self.eagle_layers = [2, L//2, L-3] # [2, 16, 29] outputs, ie. [3, L//2+1, L-2] inputs
                print(f'[Config] just set eagle_layers={self.eagle_layers}', flush=True)
            # Eagle draft must use target's rope_theta (draft config may default to wrong value)
            if self.speculate and self.draft_hf_config is not None:
                target_rope_theta = getattr(self.hf_config, 'rope_theta', 500000.0)
                draft_rope_theta = getattr(self.draft_hf_config, 'rope_theta', 10000.0)
                if target_rope_theta != draft_rope_theta:
                    print(f'[Config] Overriding eagle draft rope_theta: {draft_rope_theta} -> {target_rope_theta}', flush=True)
                    self.draft_hf_config.rope_theta = target_rope_theta
                # Also override max_position_embeddings for correct RoPE cache size
                # NOTE: Do NOT change max_model_len here - it was already correctly capped.
                # Only change draft_hf_config.max_position_embeddings for RoPE.
                target_max_pos = getattr(self.hf_config, 'max_position_embeddings', 8192)
                draft_max_pos = getattr(self.draft_hf_config, 'max_position_embeddings', 2048)
                if target_max_pos != draft_max_pos:
                    print(f'[Config] Overriding eagle draft max_position_embeddings: {draft_max_pos} -> {target_max_pos}', flush=True)
                    self.draft_hf_config.max_position_embeddings = target_max_pos
        
        assert self.max_num_batched_tokens >= self.max_model_len
