import torch
import torch.nn as nn
import torch.distributed as dist
from megatron import get_timers
from megatron import mpu, get_num_params
from megatron.model.utils import diffsort
from megatron.pruner.utils import *

module_names = ['mlp.to_4h', 'mlp.to_h', 'self_attention.dense', 'self_attention.query_key_value']

def prune_global(model, sparsity, is_sparse=False):
    print(f'[RANK {dist.get_rank()}] --> Pruning percentage: {sparsity}')

    if is_sparse:
        prune_sparse_global(model, sparsity)
    else:
        prune_dense_global(model, sparsity)

    print(f'[RANK {dist.get_rank()}] --> Finished pruning')
    dist.barrier()

def prune_dense_global(model, sparsity):
    num_gpus = mpu.get_pipeline_model_parallel_world_size()
    local_rank = mpu.get_pipeline_model_parallel_rank()

    local_num_params, num_params = get_num_params(model)
    print(f'[RANK {local_rank}] --> [BEFORE PRUNING] num local parameters: {local_num_params}, num global parameters: {num_params}')
    k = int(num_params * (1 - sparsity)) # We only need top k params (e.g. if sparsity is %80, we need %20 largest params)
    num_params_to_send = min(local_num_params, k) # If there are less than k params, send all local params
    
    disable_grad(model)

    # Flatten and concatenate weights in all local layers
    params = concat_dense_params(model)

    # Find local smallest k
    local_topk, local_topk_indices = torch.topk(torch.abs(params), num_params_to_send)

    if local_rank != 0:
        # Send size of topk to GPU 0
        dist.send(torch.cuda.LongTensor([num_params_to_send]), dst=0)

        # Send local topk to GPU 0
        dist.send(tensor=local_topk, dst=0)
    else:
        # Receive size of topks from other GPUs
        all_sizes = [torch.cuda.LongTensor([0]) for _ in range(1, num_gpus)]
        all_sizes.insert(0, torch.cuda.LongTensor([num_params_to_send]))

        for device_index in range(1, num_gpus):
            dist.recv(tensor=all_sizes[device_index], src=device_index)

        # Receive all topks from other GPUs
        all_ks = [torch.empty(all_sizes[idx].item()).cuda(local_rank) for idx in range(1, num_gpus)]
        all_ks.insert(0, local_topk)

        for device_index in range(1, num_gpus):
            dist.recv(tensor=all_ks[device_index], src=device_index)

    torch.distributed.barrier()

    if mpu.is_pipeline_first_stage():
        # Concat local topks to find global topks
        all_ks = torch.cat(all_ks)
        
        # We are interested in global topk indices
        _, indices = torch.topk(torch.abs(all_ks), k)
        #print(f'\nVALUES:{values}\n')
        #print(f'\nINDICES:{indices}\n')

        # Split the indices and send to appropriate GPU
        splitted = []
        start_tracker = 0
        for device_index in range(num_gpus):
            device_range_start = start_tracker
            device_range_end   = start_tracker + all_sizes[device_index]
            splitted.append(indices[(indices >= device_range_start) & (indices < device_range_end)])

            # print(f'all_sizes[device_index]: {all_sizes[device_index]}, splitted[device_index]: {splitted[device_index].size()[0]}')
            start_tracker += all_sizes[device_index]

        end_tracker = all_sizes[0].item()
        for device_index in range(1, num_gpus):
            # Send the size of each index tensor
            tensor_size = splitted[device_index].size()[0]
            #print_rank_0(f'Sending tensor size {tensor_size} to {device_index}')
            dist.send(tensor=torch.cuda.LongTensor([tensor_size]), dst=device_index)
            
            # Send the tensor itself
            #print_rank_0(f'Sending indices {splitted[device_index]} to {device_index} --> tracker={end_tracker}')
            dist.send(tensor=(splitted[device_index].to(torch.int32) - end_tracker), dst=device_index)

            end_tracker += all_sizes[device_index].item()
    else:
        indices_size = torch.cuda.LongTensor([0])

        # Receive the size
        dist.recv(tensor=indices_size, src=0)

        # Receive the indices
        received_indices = torch.zeros(indices_size[0].item()).cuda(local_rank).to(torch.int32)
        dist.recv(tensor=received_indices, src=0)

    torch.distributed.barrier()
    
    if mpu.is_pipeline_first_stage():
        local_indices = local_topk_indices[splitted[0]]
    else:
        local_indices = local_topk_indices[(received_indices).long()]
    
    #print(f'[RANK {local_rank}] --> Local indices picked: {local_indices}')

    state_dict = model.state_dict()
    mask = torch.ones(params.size()[0], dtype=torch.bool).cuda(local_rank)
    mask[local_indices] = False
    params.masked_fill_(mask, 0)
    print(f'[RANK {local_rank}] --> Sparsity: {(params == 0).sum().item()} / {params.numel()} = {(params == 0).sum().item() / params.numel()}')

    start = 0
    param_range_dict = get_param_range_dict(model)
    for name in param_range_dict.keys():
        shape = param_range_dict[name][0]
        size = shape[0] * shape[1]
        end = start + size
        state_dict[name] = nn.Parameter(params[start: end].reshape(shape))
        start += size

    model.load_state_dict(state_dict, strict=False)

    enable_grad(model)

    local_num_params, num_params = get_num_params(model)
    print(f'[RANK {local_rank}] --> [AFTER PRUNING] num local parameters: {local_num_params}, num global parameters: {num_params}', flush=True)

def prune_sparse_global(model, sparsity):
    timers = get_timers()
    num_gpus = mpu.get_pipeline_model_parallel_world_size()
    local_rank = mpu.get_pipeline_model_parallel_rank()

    #memory = GpuMemory(rank=local_rank)
    #print(memory)

    local_num_params, num_params = get_num_params(model)
    print(f'\n[RANK {local_rank}] --> [BEFORE PRUNING] num local parameters: {local_num_params}, num global parameters: {num_params}, ratio: {local_num_params/num_params}', flush=True)
    k = int(num_params * (1 - sparsity)) # We only need top k params (e.g. if sparsity is %80, we need %20 largest params)
    num_params_to_send = min(local_num_params, k) # If there are less than k params, send all local params

    disable_sparse_grad(model)

    timers('pruning').start()

    # 1. Flatten and concatenate csr values in all local layers
    values = concat_sparse_params(model)
    if len(values) == 0:
        values = torch.cuda.FloatTensor([])    
        column_indices = torch.cuda.LongTensor([])    
        row_offsets = torch.cuda.LongTensor([])    
    else:
        values = torch.cat(values)
        column_indices = concat_column_indices(model)
        row_offsets = concat_row_offsets(model)

    # 2. Find local k. We will keep these indices
    local_topk, local_topk_indices = torch.topk(torch.abs(values), num_params_to_send)
    #print(f'[RANK {local_rank}]: local topk: {local_topk}, local indices: {local_topk_indices}')

    if local_rank != 0:
        # Send size of topk to GPU 0
        dist.send(torch.cuda.LongTensor([num_params_to_send]), dst=0)

        # 3. Send local topk to GPU 0
        dist.send(tensor=local_topk, dst=0)
    else:
        # Receive size of topks from other GPUs
        all_sizes = [torch.cuda.LongTensor([0]) for _ in range(1, num_gpus)]
        all_sizes.insert(0, torch.cuda.LongTensor([num_params_to_send]))

        for device_index in range(1, num_gpus):
            dist.recv(tensor=all_sizes[device_index], src=device_index)

        # 3. Receive all topks from other GPUs
        all_ks = [torch.empty(all_sizes[idx].item()).cuda(local_rank) for idx in range(1, num_gpus)]
        all_ks.insert(0, local_topk)

        for device_index in range(1, num_gpus):
            dist.recv(tensor=all_ks[device_index], src=device_index)

    if mpu.is_pipeline_first_stage():
        # 4. Concat local topks to find global topks
        all_ks = torch.cat(all_ks)
        
        # 5. We are interested in global topk indices
        global_topk_values, global_topk_indices = torch.topk(torch.abs(all_ks), k)
        #print(f'\n[RANK 0] --> VALUES:{global_topk_values}\n')
        #print(f'\n[RANK 0] --> INDICES:{global_topk_indices}\n')

        # 6. Split the indices and send to appropriate GPU
        splitted = []
        start_tracker = 0
        for device_index in range(num_gpus):
            device_range_start = start_tracker
            device_range_end   = start_tracker + all_sizes[device_index]
            splitted.append(global_topk_indices[(global_topk_indices >= device_range_start) & (global_topk_indices < device_range_end)])

            start_tracker += all_sizes[device_index]

        end_tracker = all_sizes[0].item()
        for device_index in range(1, num_gpus):
            # Send the size of each index tensor
            tensor_size = splitted[device_index].size()[0]
            #print_rank_0(f'Sending tensor size {tensor_size} to {device_index}')
            dist.send(tensor=torch.cuda.LongTensor([tensor_size]), dst=device_index)
            
            # Send the tensor itself
            dist.send(tensor=(splitted[device_index].to(torch.int32) - end_tracker), dst=device_index)

            end_tracker += all_sizes[device_index].item()
    else:
        # Receive the size
        indices_size = torch.cuda.LongTensor([0])
        dist.recv(tensor=indices_size, src=0)

        # 6. Receive the indices
        received_indices = torch.zeros(indices_size[0].item()).cuda(local_rank).to(torch.int32)
        dist.recv(tensor=received_indices, src=0)

    if mpu.is_pipeline_first_stage():
        local_indices = local_topk_indices[splitted[0]]
    else:
        local_indices = local_topk_indices[(received_indices).long()]
    
    #print(f'[RANK {local_rank}] --> Local indices picked: {local_indices}')

    # 7. Prepare the mask to remove smallest elements
    csr_dict = dict()
    mask_all = torch.zeros(values.size()[0], dtype=torch.bool).cuda(local_rank) # Remove all
    mask_all[local_indices] = True # Keep largest local indices

    start = 0
    param_size_dict = get_sparse_param_sizes(model)
    for index, name in enumerate(param_size_dict.keys()):
        size = param_size_dict[name]

        values_with_zeros = values[start: start + size].detach().clone()
        column_indices_with_zeros = column_indices[start: start + size].detach().clone()
        offsets = row_offsets[index]
        csr_offsets = offsets.detach().clone()
        
        mask = mask_all[start: start + size]

        start += size

        # Number of elements we keep must be multiple of 4 due to Sputnik constraint.
        idx = 0
        while mask.sum().item() % 4 != 0:
            if mask[idx] == False:
                mask[idx] = True
            idx += 1

        # New csr values
        csr_values = torch.masked_select(values_with_zeros, mask)
        csr_dict[name] = nn.Parameter(csr_values)

        # New column indices
        csr_column_indices = torch.masked_select(column_indices_with_zeros, mask)
        csr_dict[name[:-6] + "column_indices"] = csr_column_indices

        # New offsets
        cumsum = 0
        for offset in range(1, offsets.size()[0]):
            offset_start = offsets[offset - 1].item()
            offset_end = offsets[offset].item()

            cumsum += (mask[offset_start:offset_end] == True).sum().item()
            csr_offsets[offset] = cumsum

        csr_dict[name[:-6] + "row_offsets"] = csr_offsets

        # New row_indices
        csr_dict[name[:-6] + "row_indices"] = diffsort(csr_offsets)

    # 8. Update values, column_indices, row_offsets, and row_indices
    for name, module in model.named_modules():
        if any(prune_module_name in name for prune_module_name in module_names):
            #print(f'[RANK {local_rank}] --> {name}: [BEFORE] values {module.values.size()}, column_indices {module.column_indices.size()}, row_offsets: {module.row_offsets.size()}')
            #print(f'[RANK {local_rank}] --> {name}: [BEFORE] values {module.values}, column_indices {module.column_indices}, row_offsets: {module.row_offsets}')
            module.values = csr_dict[name[7:] + ".values"]
            module.column_indices = csr_dict[name[7:] + ".column_indices"]
            module.row_offsets = csr_dict[name[7:] + ".row_offsets"]
            module.row_indices = csr_dict[name[7:] + ".row_indices"]
            #print(f'[RANK {local_rank}] --> {name}: [AFTER] values {module.values.size()}, column_indices {module.column_indices.size()}, row_offsets: {module.row_offsets.size()}')
            #print(f'[RANK {local_rank}] --> {name}: [AFTER] values {module.values}, column_indices {module.column_indices}, row_offsets: {module.row_offsets}')
    
    timers('pruning').stop()
    pruning_time = timers('pruning').elapsed()
    print(f'[RANK {torch.distributed.get_rank()}] --> Pruning took: {pruning_time * 1000} ms')

    enable_sparse_grad(model)

    #local_num_params, num_params = get_num_params(model)
    #print(f'\n[RANK {local_rank}] --> [AFTER PRUNING] num local parameters: {local_num_params}, num global parameters: {num_params}, ratio: {local_num_params/num_params}', flush=True)
