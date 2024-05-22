import torch
from megatron import get_timers
from megatron.global_vars import get_args
from .deepspeed import deepspeed_balance
from .deepspeed_param import deepspeed_param_balance
from .diffusion import diffusion_balance
from .diffusion_param import diffusion_param_balance
from megatron.balancer.pack import pack_model

def load_balance(model, balancer):
    max_memory_allocated = torch.cuda.max_memory_allocated(torch.distributed.get_rank())
    
    timers = get_timers()
    timers('load-balance').start()
    
    if balancer in ["deepspeed_static", "deepspeed_param"]:
        global_memory_usages = deepspeed_param_balance(model, max_memory_allocated)
    elif balancer == "deepspeed":
        global_memory_usages = deepspeed_balance(model, max_memory_allocated)
    elif balancer == "diffusion":
        global_memory_usages = diffusion_balance(model, max_memory_allocated)
    elif balancer == "diffusion_param":
        global_memory_usages = diffusion_param_balance(model, max_memory_allocated)
    else:
        raise ValueError(f'Invalid balancer: {balancer}')

    timers('load-balance').stop()
    elapsed = timers('load-balance').elapsed()
    print(f'[RANK {torch.distributed.get_rank()}] --> Load balancing took: {elapsed * 1000} ms')

    torch.distributed.barrier()

    return global_memory_usages
