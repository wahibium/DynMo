import torch
import torch.distributed as dist
from megatron import mpu
from megatron import get_args, get_num_params
from megatron.balancer.utils import get_layer_times, get_memory_usage_per_layer, prefix_sum, cumsum
from bisect import bisect_left
from megatron.balancer.transfer import Transfer, perform_transfers

MAX_MEMORY = 32_000_000_000 # 40 GB with fragmentation

def _lprobe(times, num_parts, bottleneck):
    num_items = len(times)
    total_weight = times[-1]

    # initialize partitioning
    parts = [0] * (num_parts + 1)
    for p in range(1, num_parts + 1):
        parts[p] = num_items

    bsum = bottleneck  # running sum of target weight for pth partition
    chunksize = num_items // num_parts
    step = chunksize
    for p in range(1, num_parts):
        # Jump to the next bucket
        while (step < num_items) and (times[step] < bsum):
            step += chunksize

        # Find the end index of partition p
        parts[p] = bisect_left(times,
                               bsum,
                               lo=step - chunksize,
                               hi=min(step,
                                      num_items))
        # Nothing more to partition, return early
        if parts[p] == num_items:
            # See if the current partition is overweight.
            part_size = times[-1] - times[parts[p - 1]]
            return parts, part_size < bottleneck

        # Next partition target
        bsum = times[parts[p] - 1] + bottleneck

    return parts, bsum >= total_weight

def _rb_partition_balanced(times, num_parts, eps):
    total_weight = times[-1]
    lower = total_weight / num_parts  # best case heaviest partition
    upper = total_weight  # worst case heaviest partition

    # Do a binary search for the best partitioning
    while upper > lower + eps:
        mid = lower + ((upper - lower) / 2)
        parts, success = _lprobe(times, num_parts, mid)
        if success:
            upper = mid
        else:
            lower = mid + eps
    return upper

def partition_balanced(times, num_parts, eps=1e-3):
    """ Partitioning algorithm from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/utils.py """
    times_ = prefix_sum(times)

    # Find the smallest bottleneck (time of heaviest partition)
    bottleneck = _rb_partition_balanced(times_, num_parts, eps=eps)

    # Now compute that partitioning
    parts, success = _lprobe(times_, num_parts, bottleneck)
    assert success

    return parts

def find_src(current_ranges, element):
    print(f'[find_src] --> current_ranges: {current_ranges}, element: {element}')
    for range_idx, current_range in enumerate(current_ranges):
        if element in current_range:
            layer_idx = current_range.index(element)
            return range_idx, layer_idx

def setup_ranges(partitions, cum_num_layers):
    suggested_ranges = []
    current_ranges   = []
    for idx in range(1, len(partitions)):
        suggested_start = partitions[idx - 1]
        suggested_end = partitions[idx]

        current_start = cum_num_layers[idx - 1]
        current_end = cum_num_layers[idx]

        suggested_range = list(range(suggested_start, suggested_end))
        current_range   = list(range(current_start, current_end))

        suggested_ranges.append(suggested_range)
        current_ranges.append(current_range)

    print(f'Current ranges: {current_ranges}, Suggested ranges: {suggested_ranges}')
    return current_ranges, suggested_ranges

def get_transfers(partitions, global_num_layers, global_memory_usages):
    cum_num_layers = cumsum(global_num_layers)
    print(f'cumsum: {cum_num_layers}, partitions: {partitions}')
    transfers = []

    memory_per_layer = get_memory_usage_per_layer(global_memory_usages, global_num_layers)
    current_ranges, suggested_ranges = setup_ranges(partitions, cum_num_layers)

    # Calculate transfers
    for gpu_idx, (suggested_range, current_range) in enumerate(zip(suggested_ranges, current_ranges)):
        for element in suggested_range:
            if element not in current_range:
                src, layer_idx = find_src(current_ranges, element)

                #print(f'Src: {src}, dst: {gpu_idx}, dst memory: {(global_memory_usages[gpu_idx] / (1024 * 1024)):.2f}, src mem per layer: {(memory_per_layer[src] / (1024 * 1024)):.2f}')

                if global_memory_usages[gpu_idx] + memory_per_layer[src] > MAX_MEMORY:
                    print(f'SKIPPING!!!!!!!!')
                    continue

                global_memory_usages[gpu_idx] += memory_per_layer[src]
                global_memory_usages[src] -= memory_per_layer[src]

                global_num_layers[gpu_idx] += 1
                global_num_layers[src] -= 1
                transfers.append(Transfer(src=src, dst=gpu_idx, layer_idx=layer_idx))

    print(f'Transfers: {transfers}')
    print(f'Max mem usage prediction: {global_memory_usages}')

    return transfers

def deepspeed_balance(model, max_memory_allocated):
    num_gpus = mpu.get_pipeline_model_parallel_world_size()
    local_rank = mpu.get_pipeline_model_parallel_rank()
    args = get_args()

    # Print number of local and global parameters before balancing
    local_num_params, num_params = get_num_params(model)
    print(f'\n[RANK {local_rank}] --> [BEFORE BALANCING] num local parameters: {local_num_params}, num global parameters: {num_params}, ratio: {local_num_params/num_params}', flush=True)

    # 1. Gather all layer local_times to RANK 0
    layer_numbers = [layer.layer_number - 1 for layer in model.module.language_model.encoder.layers]
    print(f'\n[RANK {local_rank}] --> [BEFORE BALANCING] layer_numbers: {layer_numbers}', flush=True)
    local_times = get_layer_times(layer_numbers, args)
    global_memory_usages = None
    
    if not mpu.is_pipeline_first_stage():
        # 1.1 Send number of layers
        dist.send(tensor=torch.cuda.LongTensor([len(layer_numbers)]), dst=0)

        # 1.2 Send layer local_times
        dist.send(tensor=torch.tensor(local_times).cuda(local_rank), dst=0)

        # 1.3 Send memory usage
        dist.send(tensor=torch.cuda.LongTensor([max_memory_allocated]), dst=0)
    else:
        # 1.1 Receive num layers
        global_num_layers = [torch.cuda.LongTensor([0]) for _ in range(1, num_gpus)]
        global_num_layers.insert(0, torch.cuda.LongTensor([len(layer_numbers)])) 

        for device_index in range(1, num_gpus):
            dist.recv(tensor=global_num_layers[device_index], src=device_index)

        # 1.2 Receive all layer local_times
        global_times = [torch.tensor(local_times).cuda(local_rank)]
        for idx in range(1, num_gpus):
            if idx == num_gpus - 1:
                global_times.append(torch.empty(global_num_layers[idx].item() + 2).cuda(local_rank))
            else:
                global_times.append(torch.empty(global_num_layers[idx].item()).cuda(local_rank))

        for device_index in range(1, num_gpus):
            dist.recv(tensor=global_times[device_index], src=device_index)
        
        print(f'[RANK 0] --> global_times: {global_times}')

        # 1.3 Receive all memory usages
        global_memory_usages = [torch.cuda.LongTensor([0]) for _ in range(1, num_gpus)]
        global_memory_usages.insert(0, torch.cuda.LongTensor([max_memory_allocated]))

        for device_index in range(1, num_gpus):
            dist.recv(tensor=global_memory_usages[device_index], src=device_index)

        print(f'Memory usages: {global_memory_usages}')

    # 2. Partition layers
    if mpu.is_pipeline_first_stage():
        global_times = torch.cat(global_times).tolist()

        # [0] --> embedding, layer_{0..N}
        # [1] --> layer_{0..N}, [2] --> layer_{0..N}
        # [3] --> layer_{0..N}, pooler, post_language_model_processing
        print(f'[RANK 0] --> TIMES: {global_times}', flush=True)
        partitions = partition_balanced(global_times, num_gpus)

        for idx in range(1, len(partitions)): # remove embedding
            partitions[idx] -= 1
        partitions[-1] -= 2 # remove pooler and postprocessing

        for idx in range(1, len(partitions)):
            if partitions[-idx - 1] > partitions[-idx]:
                partitions[-idx - 1] = partitions[-idx]

        print(f'PARTITIONS: {partitions}', flush=True)

        print(f'global_num_layers: {global_num_layers}')
        global_num_layers = [num_layers.item() for num_layers in global_num_layers]
        global_memory_usages = [memory.item() for memory in global_memory_usages]

        transfers = get_transfers(partitions, global_num_layers, global_memory_usages)

        num_transfer_size = [0 for _ in range(num_gpus)]
        all_transfers = [[] for _ in range(num_gpus)]

        for transfer in transfers:
            num_transfer_size[transfer.src] += 1
            num_transfer_size[transfer.dst] += 1
            # (layer_idx, pair_rank, comm_type)
            all_transfers[transfer.src].append([transfer.layer_idx, transfer.dst, 0])
            all_transfers[transfer.dst].append([transfer.layer_idx, transfer.src, 1])

        my_transfers = all_transfers[0]

        for device_idx in range(1, num_gpus):
            dist.send(tensor=torch.cuda.LongTensor([num_transfer_size[device_idx]]), dst=device_idx)
            dist.send(tensor=torch.cuda.LongTensor(all_transfers[device_idx]), dst=device_idx)
    else:
        num_transfer_size = torch.cuda.LongTensor([0])
        dist.recv(tensor=num_transfer_size, src=0)

        my_transfers = torch.cuda.LongTensor([[0, 0, 0] for _ in range(num_transfer_size)])
        dist.recv(tensor=my_transfers, src=0)

        my_transfers = my_transfers.tolist()
    
    dist.barrier()
    
    perform_transfers(model, my_transfers, local_rank, args, "deepspeed")
    
    layer_numbers = [layer.layer_number - 1 for layer in model.module.language_model.encoder.layers]
    print(f'\n[RANK {local_rank}] --> [AFTER BALANCING] layer_numbers: {layer_numbers}')
    #local_num_params, num_params = get_num_params(model)
    #print(f'\n[RANK {local_rank}] --> [AFTER BALANCING] num local parameters: {local_num_params}, num global parameters: {num_params}, ratio: {local_num_params/num_params}', flush=True)

    return global_memory_usages