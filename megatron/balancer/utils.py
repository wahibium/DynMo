import torch
from megatron import get_timers
from megatron import mpu

def cumsum(elements):
    return [0] + prefix_sum(elements)

def prefix_sum(times):
    """ Compute an inclusive prefix sum.
    Example:
        >>> prefix_sum([3,4,5])
        [3, 7, 12]
    """
    times_ = [w for w in times]
    for x in range(1, len(times_)):
        times_[x] += times_[x - 1]
    return times_

def get_layer_times(layer_numbers, args):
    timers = get_timers()

    # Get layer compute times
    timers_to_log = []
    def add_to_timer(name):
        if name in timers.timers:
            timers_to_log.append(name)

    if mpu.is_pipeline_first_stage():
        add_to_timer('embedding')

    for i in layer_numbers:
        add_to_timer(f'layer_{i}')

    if mpu.is_pipeline_last_stage():
        add_to_timer('pooler')
        add_to_timer('post_language_model_processing')

    compute_times = timers.get_times(timers_to_log, normalizer=args.log_interval, reset=True)

    print(f'[RANK {torch.distributed.get_rank()}] --> Compute times: {compute_times}')

    return compute_times

def get_layer_parameters(model, local_rank, num_gpus):
    dense_module_names  = ['word_embeddings', 'position_embeddings', 'pooler.dense', 'lm_head.dense']

    layer_params = []

    for module in model.module.language_model.encoder.layers:
        num_params = 0
        for name, param in module.named_parameters():
            num_params += param.numel()
            #print(f'[RANK {local_rank}] --> Added parameter: {name}', flush=True)
        
        layer_params.append(num_params)
    
    for module_name, module in model.named_modules():
        if any(prune_module_name in module_name for prune_module_name in dense_module_names):
            num_params = 0
            for param_name, param in module.named_parameters():
                num_params += param.numel()
                # print(f'[RANK {local_rank}] --> Added parameter {module_name}.{param_name}', flush=True)
            
            if local_rank == num_gpus - 1:
                layer_params.append(num_params)
            elif local_rank == 0:
                layer_params.insert(0, num_params)

    print(f'[RANK {local_rank}] --> Before layer_params: {layer_params}')
    if local_rank == 0:
        layer_params[1] += layer_params[0]
        layer_params = layer_params[1:]
    elif local_rank == num_gpus - 1:
        layer_params[-2] += layer_params[-1]
        layer_params = layer_params[:-1]
    print(f'[RANK {local_rank}] --> After layer_params: {layer_params}')

    print(f'[RANK {local_rank}] --> len: {len(layer_params)}, total parameters: {sum(layer_params)}')
    return layer_params

def get_memory_usage_per_layer(global_memory_usages, global_num_layers):
    memory_per_layer = [0 for _ in range(len(global_memory_usages))]

    for idx, _ in enumerate(range(len(global_memory_usages))):
        if global_num_layers[idx] == 0:
            memory_per_layer[idx] = global_memory_usages[idx]
        else:
            if idx == 0:
                memory_per_layer[idx] = global_memory_usages[idx] // (global_num_layers[idx] + 1)
            else:
                memory_per_layer[idx] = global_memory_usages[idx] // global_num_layers[idx]

    print(f'Memory per layer: {memory_per_layer}')
    return memory_per_layer