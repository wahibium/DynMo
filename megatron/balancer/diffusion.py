import torch
import torch.distributed as dist
from megatron import mpu
from megatron import get_args, get_num_params
from megatron.balancer.utils import get_layer_times, get_memory_usage_per_layer
import statistics
from copy import deepcopy
from megatron.balancer.transfer import Transfer, perform_transfers

MAX_MEMORY = 32_000_000_000 # 40 GB with fragmentation

def find_lightest_layer(times, src, num_gpus):
    print(f'Finding lightest layer in GPU {src}: {times}')
    lightest_idx = -1
    if src == 0: # do not take embedding into account
        local_times = times[src][1:]
        lightest_idx = local_times.index(min(local_times))
    elif src == num_gpus - 1: # do not take post processing into account
        local_times = times[src][:-2]
        lightest_idx = local_times.index(min(local_times))
    else:
        lightest_idx = times[src].index(min(times[src]))

    assert lightest_idx != -1
    return lightest_idx

def get_transfers(times, num_gpus, global_memory_usages):
    global_num_layers = [len(time) for time in times]
    final_times = deepcopy(times)
    max_iter = 5
    best_variance = float("inf")
    transfers = []

    memory_per_layer = get_memory_usage_per_layer(global_memory_usages, global_num_layers)

    for iteration in range(max_iter):
        print("-" * 10)
        # Calculate load of each GPU
        loads = [sum(time) for time in final_times]
        avg_load = sum(loads) / num_gpus
        variance = statistics.variance(loads)
        status = ["Overloaded" if load > avg_load else "Underloaded" for load in loads]
        
        print(f'\n[Iter {iteration}] --> loads: {loads}, variance: {variance} status: {status}, times: {times}', flush=True)

        for src in range(num_gpus):
            if status[src] == "Overloaded":
                if src == 0 and len(times[src]) == 1:
                    continue # only embedding left
                if src == num_gpus - 1 and len(times[src]) == 2:    
                    continue # only postprocessing left

                # Find the lightest gpu
                dst = loads.index(min(loads))
                
                if status[dst] == "Underloaded":
                    # Send the lightest layer
                    lightest_idx = find_lightest_layer(times, src, num_gpus)

                    if global_memory_usages[dst] + memory_per_layer[src] > MAX_MEMORY:
                        print(f'Not enough memory on GPU {dst}')
                        continue

                    if src == 0:
                        if dst == num_gpus - 1:
                            # if dst is last rank, insert because there are post processing units at the end.
                            times[dst].insert(0, times[src][lightest_idx])
                        else:
                            times[dst].append(times[src][lightest_idx])

                        del times[src][lightest_idx]
                    else:
                        times[dst].append(times[src][lightest_idx])
                        del times[src][lightest_idx]

                    print(f'GPU {src} trying to send layer {lightest_idx} to GPU {dst} ')

                    ## CHECK IF THIS TRANSFER DECREASES THE VARIANCE. IF NOT REJECT
                    
                    # Calculate new loads
                    new_loads = [sum(time) for time in times]
                    avg_load = sum(new_loads) / num_gpus

                     # Calculate new variance
                    new_variance = statistics.variance(new_loads)

                    # New status
                    new_status = ["Overloaded" if load > avg_load else "Underloaded" for load in loads]
                    print(f'[Iter {iteration}] --> new loads: {new_loads}, new variance: {new_variance} new status: {new_status}', flush=True)

                    if new_variance < best_variance:
                        best_variance = new_variance
                        final_times = deepcopy(times)
                        #print([len(time) for time in final_times])
                    else:
                        print(f'GPU {src} : {lightest_idx} -> GPU {dst}: Rejected ')
                        times = deepcopy(final_times)
                        continue

                    print(f'GPU {src} : {lightest_idx} -> GPU {dst}: Accepted ')
                    global_memory_usages[dst] += memory_per_layer[src]
                    global_memory_usages[src] -= memory_per_layer[src]

                    transfers.append(Transfer(src, dst, lightest_idx))
        print("-" * 10)

    print(f'Final times: {[len(time) for time in final_times]}, Transfers: {transfers}')
    
    return transfers

def diffusion_balance(model, max_memory_allocated):
    num_gpus = mpu.get_pipeline_model_parallel_world_size()
    local_rank = mpu.get_pipeline_model_parallel_rank()
    args = get_args()

    local_num_params, num_params = get_num_params(model)
    print(f'\n[RANK {local_rank}] --> [BEFORE BALANCING] num local parameters: {local_num_params}, num global parameters: {num_params}, ratio: {local_num_params/num_params}', flush=True)

    layer_numbers = [layer.layer_number - 1 for layer in model.module.language_model.encoder.layers]
    print(f'\n[RANK {local_rank}] --> [BEFORE BALANCING] layer_numbers: {layer_numbers}', flush=True)

    # 1. All reduce all layer times
    # 1.1 Get local layer times
    local_layer_times = get_layer_times(layer_numbers, args)
    num_layer_times = len(local_layer_times)
    global_memory_usages = None

    if not mpu.is_pipeline_first_stage():
        # 1.1 Send number of layers
        dist.send(tensor=torch.cuda.LongTensor([num_layer_times]), dst=0)

        # 1.2 Send layer local_times
        dist.send(tensor=torch.tensor(local_layer_times).cuda(local_rank), dst=0)

        # 1.3 Send memory usage
        dist.send(tensor=torch.cuda.LongTensor([max_memory_allocated]), dst=0)
    else:
        # 1.1 Receive num layers
        all_num_layers = [torch.cuda.LongTensor([0]) for _ in range(1, num_gpus)]
        all_num_layers.insert(0, torch.cuda.LongTensor([num_layer_times])) 

        for device_index in range(1, num_gpus):
            dist.recv(tensor=all_num_layers[device_index], src=device_index)

        # 1.2 Receive all layer local_times
        global_times = [torch.empty(all_num_layers[idx].item()).cuda(local_rank) for idx in range(1, num_gpus)]
        global_times.insert(0, torch.tensor(local_layer_times).cuda(local_rank))

        for device_index in range(1, num_gpus):
            dist.recv(tensor=global_times[device_index], src=device_index)

        global_times = [time.tolist() for time in global_times]

        # 1.3 Receive all memory usages
        global_memory_usages = [torch.cuda.LongTensor([0]) for _ in range(1, num_gpus)]
        global_memory_usages.insert(0, torch.cuda.LongTensor([max_memory_allocated]))

        for device_index in range(1, num_gpus):
            dist.recv(tensor=global_memory_usages[device_index], src=device_index)

        print(f'Memory usages: {global_memory_usages}')
    
    if mpu.is_pipeline_first_stage():
        global_memory_usages = [memory.item() for memory in global_memory_usages]
        transfers = get_transfers(global_times, num_gpus, global_memory_usages)
        
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
        
    print(f'[RANK {local_rank}] --> my_transfers: {my_transfers}', flush=True)

    perform_transfers(model, my_transfers, local_rank, args, "diffusion")

    layer_numbers = [layer.layer_number - 1 for layer in model.module.language_model.encoder.layers]
    print(f'\n[RANK {local_rank}] --> [AFTER BALANCING] layer_numbers: {layer_numbers}')
    #local_num_params, num_params = get_num_params(model)
    #print(f'\n[RANK {local_rank}] --> [AFTER BALANCING] num local parameters: {local_num_params}, num global parameters: {num_params}, ratio: {local_num_params/num_params}', flush=True)

    return global_memory_usages