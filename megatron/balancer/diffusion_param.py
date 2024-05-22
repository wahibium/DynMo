import torch
import torch.distributed as dist
from megatron import mpu
from megatron import get_args, get_num_params
from megatron.balancer.utils import get_layer_parameters
from megatron.balancer.transfer import perform_transfers
from megatron.balancer.diffusion import get_transfers

def diffusion_param_balance(model, max_memory_allocated):
    num_gpus = mpu.get_pipeline_model_parallel_world_size()
    local_rank = mpu.get_pipeline_model_parallel_rank()
    args = get_args()

    local_num_params, num_params = get_num_params(model)
    print(f'\n[RANK {local_rank}] --> [BEFORE BALANCING] num local parameters: {local_num_params}, num global parameters: {num_params}, ratio: {local_num_params/num_params}', flush=True)

    layer_numbers = [layer.layer_number - 1 for layer in model.module.language_model.encoder.layers]
    print(f'\n[RANK {local_rank}] --> [BEFORE BALANCING] layer_numbers: {layer_numbers}')

    # 1. All reduce all layer times
    # 1.1 Get local layer times
    local_layer_params = get_layer_parameters(model, local_rank, num_gpus)
    global_memory_usages = None

    if not mpu.is_pipeline_first_stage():
        # 1.1 Send number of layers
        dist.send(tensor=torch.cuda.LongTensor([len(local_layer_params)]), dst=0)

        # 1.2 Send layer local_times
        dist.send(tensor=torch.cuda.LongTensor(local_layer_params), dst=0)

        # 1.3 Send memory usage
        dist.send(tensor=torch.cuda.LongTensor([max_memory_allocated]), dst=0)
    else:
        # 1.1 Receive num layers
        all_num_layers = [torch.cuda.LongTensor([0]) for _ in range(1, num_gpus)]
        all_num_layers.insert(0, torch.cuda.LongTensor([len(local_layer_params)])) 

        for device_index in range(1, num_gpus):
            dist.recv(tensor=all_num_layers[device_index], src=device_index)

        # 1.2 Receive all layer local_times
        global_params = [torch.empty(all_num_layers[idx].item(), dtype=torch.long).cuda(local_rank) for idx in range(1, num_gpus)]
        global_params.insert(0, torch.cuda.LongTensor(local_layer_params))

        for device_index in range(1, num_gpus):
            dist.recv(tensor=global_params[device_index], src=device_index)

        # 1.3 Receive all memory usages
        global_memory_usages = [torch.cuda.LongTensor([0]) for _ in range(1, num_gpus)]
        global_memory_usages.insert(0, torch.cuda.LongTensor([max_memory_allocated]))

        for device_index in range(1, num_gpus):
            dist.recv(tensor=global_memory_usages[device_index], src=device_index)

        print(f'Memory usages: {global_memory_usages}')

    if mpu.is_pipeline_first_stage():
        global_params = [time.tolist() for time in global_params]
        global_memory_usages = [memory.item() for memory in global_memory_usages]
        transfers = get_transfers(global_params, num_gpus, global_memory_usages)
        
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

    print(f'[RANK {local_rank}] --> my_transfers: {my_transfers}', flush=True)

    perform_transfers(model, my_transfers, local_rank, args, "diffusion_param")

    layer_numbers = [layer.layer_number - 1 for layer in model.module.language_model.encoder.layers]
    print(f'\n[RANK {local_rank}] --> [AFTER BALANCING] layer_numbers: {layer_numbers}')
    #local_num_params, num_params = get_num_params(model)
    #print(f'\n[RANK {local_rank}] --> [AFTER BALANCING] num local parameters: {local_num_params}, num global parameters: {num_params}, ratio: {local_num_params/num_params}', flush=True)

    return global_memory_usages