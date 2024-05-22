import torch
import torch.distributed as dist
from megatron import mpu
from megatron import get_args, get_num_params
from megatron.balancer.transfer import perform_transfers
from megatron.balancer.utils import get_layer_parameters
from megatron.balancer.deepspeed import partition_balanced, get_transfers

def deepspeed_param_balance(model, max_memory_allocated):
    num_gpus = mpu.get_pipeline_model_parallel_world_size()
    local_rank = mpu.get_pipeline_model_parallel_rank()
    args = get_args()

    # Print number of local and global parameters before balancing
    local_num_params, num_params = get_num_params(model)
    print(f'\n[RANK {local_rank}] --> [BEFORE BALANCING] num local parameters: {local_num_params}, num global parameters: {num_params}, ratio: {local_num_params/num_params}', flush=True)

    # 1. Gather all layer local_num_params to RANK 0
    layer_numbers = [layer.layer_number - 1 for layer in model.module.language_model.encoder.layers]
    print(f'\n[RANK {local_rank}] --> [BEFORE BALANCING] layer_numbers: {layer_numbers}', flush=True)

    local_num_params = get_layer_parameters(model, local_rank, num_gpus)
    global_memory_usages = None
    
    if not mpu.is_pipeline_first_stage():
        # 1.1 Send number of layers
        dist.send(tensor=torch.cuda.LongTensor([len(layer_numbers)]), dst=0)

        # 1.2 Send layer local_num_params
        dist.send(tensor=torch.cuda.LongTensor(local_num_params), dst=0)

        # 1.3 Send memory usage
        dist.send(tensor=torch.cuda.LongTensor([max_memory_allocated]), dst=0)
    else:
        # 1.1 Receive num layers
        global_num_layers = [torch.cuda.LongTensor([0]) for _ in range(1, num_gpus)]
        global_num_layers.insert(0, torch.cuda.LongTensor([len(layer_numbers)])) 

        for device_index in range(1, num_gpus):
            dist.recv(tensor=global_num_layers[device_index], src=device_index)

        # 1.2 Receive all layer local_num_params
        global_params = [torch.cuda.LongTensor(local_num_params)]
        for idx in range(1, num_gpus):
            if idx == num_gpus - 1:
                global_params.append(torch.empty(global_num_layers[idx].item() + 2, dtype=torch.long).cuda(local_rank))
            else:
                global_params.append(torch.empty(global_num_layers[idx].item(), dtype=torch.long).cuda(local_rank))

        for device_index in range(1, num_gpus):
            #print(f'[RANK 0] --> Before receiving from index {device_index}: {global_params[device_index]}')
            dist.recv(tensor=global_params[device_index], src=device_index)
            #print(f'[RANK 0] --> After receiving from index {device_index}: {global_params[device_index]}')

        print(f'[RANK 0] --> global_params: {global_params}')

        # 1.3 Receive all memory usages
        global_memory_usages = [torch.cuda.LongTensor([0]) for _ in range(1, num_gpus)]
        global_memory_usages.insert(0, torch.cuda.LongTensor([max_memory_allocated]))

        for device_index in range(1, num_gpus):
            dist.recv(tensor=global_memory_usages[device_index], src=device_index)

        print(f'Memory usages: {global_memory_usages}')

    # 2. Partition layers
    if mpu.is_pipeline_first_stage():
        global_params = torch.cat(global_params).tolist()

        # [0] --> embedding, layer_{0..N}
        # [1] --> layer_{0..N}, [2] --> layer_{0..N}
        # [3] --> layer_{0..N}, pooler, post_language_model_processing
        partitions = partition_balanced(global_params, num_gpus)

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

        # Prepare all transfers
        num_transfer_size = [0 for _ in range(num_gpus)]
        all_transfers = [[] for _ in range(num_gpus)]

        for transfer in transfers:
            num_transfer_size[transfer.src] += 1
            num_transfer_size[transfer.dst] += 1
            # (layer_idx, pair_rank, comm_type)
            all_transfers[transfer.src].append([transfer.layer_idx, transfer.dst, 0])
            all_transfers[transfer.dst].append([transfer.layer_idx, transfer.src, 1])

        my_transfers = all_transfers[0]

        # Scatter transfers
        for device_idx in range(1, num_gpus):
            dist.send(tensor=torch.cuda.LongTensor([num_transfer_size[device_idx]]), dst=device_idx)
            dist.send(tensor=torch.cuda.LongTensor(all_transfers[device_idx]), dst=device_idx)
    else:
        # Get number of transfers
        num_transfer_size = torch.cuda.LongTensor([0])
        dist.recv(tensor=num_transfer_size, src=0)

        # Get transfers
        my_transfers = torch.cuda.LongTensor([[0, 0, 0] for _ in range(num_transfer_size)])
        dist.recv(tensor=my_transfers, src=0)

        my_transfers = my_transfers.tolist()

    dist.barrier()

    perform_transfers(model, my_transfers, local_rank, args, "deepspeed_param")

    layer_numbers = [layer.layer_number - 1 for layer in model.module.language_model.encoder.layers]
    print(f'\n[RANK {local_rank}] --> [AFTER BALANCING] layer_numbers: {layer_numbers}')
    #local_num_params, num_params = get_num_params(model)
    #print(f'\n[RANK {local_rank}] --> [AFTER BALANCING] num local parameters: {local_num_params}, num global parameters: {num_params}, ratio: {local_num_params/num_params}', flush=True)

    return global_memory_usages