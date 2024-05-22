import torch
import torch.distributed as dist
from megatron import get_args
from megatron import mpu
from megatron.balancer.transfer import Transfer, perform_transfers
from megatron.global_vars import get_active_gpus

MAX_MEMORY = 32_000_000_000 # 40 GB with fragmentation

def pack_model(model, global_memory_usages):
    transfers = []
    num_gpus = mpu.get_pipeline_model_parallel_world_size()
    local_rank = mpu.get_pipeline_model_parallel_rank()
    args = get_args()
    active_gpus = get_active_gpus()
    max_memory_allocated = torch.cuda.max_memory_allocated(local_rank)

    layer_numbers = [layer.layer_number - 1 for layer in model.module.language_model.encoder.layers]
    print(f'\n[RANK {local_rank}] --> [BEFORE PACKING] layer_numbers: {layer_numbers}', flush=True)

    if not mpu.is_pipeline_first_stage():
        # 1.1 Send number of layers
        dist.send(tensor=torch.cuda.LongTensor([len(layer_numbers)]), dst=0)

        # 1.2 Send memory usage
        dist.send(tensor=torch.cuda.LongTensor([max_memory_allocated]), dst=0)
    else:
        # 1.1 Receive num layers
        global_num_layers = [torch.cuda.LongTensor([0]) for _ in range(1, num_gpus)]
        global_num_layers.insert(0, torch.cuda.LongTensor([len(layer_numbers)])) 

        for device_index in range(1, num_gpus):
            dist.recv(tensor=global_num_layers[device_index], src=device_index)

        # 1.2 Receive all memory usages
        memory_usages = [torch.cuda.LongTensor([0]) for _ in range(1, num_gpus)]
        memory_usages.insert(0, torch.cuda.LongTensor([max_memory_allocated]))

        for device_index in range(1, num_gpus):
            dist.recv(tensor=memory_usages[device_index], src=device_index)

        if global_memory_usages is not None:
            memory_usages = global_memory_usages
        else:
            memory_usages = [memory.item() for memory in memory_usages]
        
        global_num_layers = [num_layers.item() for num_layers in global_num_layers]

        print(f'Memory usages: {memory_usages}')

    if mpu.is_pipeline_first_stage():
        print(f'Active gpus before packing: {active_gpus}, memory usages before packing: {memory_usages}')
        # 1.3 Pack
        for src in range(num_gpus):
            for dst in range(src + 1, num_gpus):
                num_active_gpus = sum(active_gpus)
                if memory_usages[src] + memory_usages[dst] < MAX_MEMORY and num_active_gpus > args.packed_num_gpus:
                    active_gpus[src] = 0

                    for layer_idx in range(global_num_layers[src]):
                        transfers.append(Transfer(src, dst, layer_idx))

                    # Update memory usages
                    memory_usages[dst] += memory_usages[src]
                    memory_usages[src] = 0

                    # Update number of layers
                    global_num_layers[dst] += global_num_layers[src]
                    global_num_layers[src] = 0

        num_transfer_size = [0 for _ in range(num_gpus)]
        all_transfers = [[] for _ in range(num_gpus)]

        if sum(active_gpus) == args.packed_num_gpus:
            for transfer in transfers:
                num_transfer_size[transfer.src] += 1
                num_transfer_size[transfer.dst] += 1
                # (layer_idx, pair_rank, comm_type)
                all_transfers[transfer.src].append([transfer.layer_idx, transfer.dst, 0])
                all_transfers[transfer.dst].append([transfer.layer_idx, transfer.src, 1])

            print(f'PACKED!!! Active gpus: {active_gpus}, New memory usages: {memory_usages}')
        else:
            print(f'Cannot pack!!! Active gpus: {active_gpus}, Memory usages: {memory_usages}')

        my_transfers = all_transfers[0]

        for device_idx in range(1, num_gpus):
            # Send number of transfers
            dist.send(tensor=torch.cuda.LongTensor([num_transfer_size[device_idx]]), dst=device_idx)
            # Send transfers
            dist.send(tensor=torch.cuda.LongTensor(all_transfers[device_idx]), dst=device_idx)
    else:
        num_transfer_size = torch.cuda.LongTensor([0])
        # Recv number of transfers
        dist.recv(tensor=num_transfer_size, src=0)

        my_transfers = torch.cuda.LongTensor([[0, 0, 0] for _ in range(num_transfer_size)])
        # Recv transfers
        dist.recv(tensor=my_transfers, src=0)

        my_transfers = my_transfers.tolist()

    dist.barrier()
    perform_transfers(model, my_transfers, local_rank, args, "deepspeed")
    
    layer_numbers = [layer.layer_number - 1 for layer in model.module.language_model.encoder.layers]
    print(f'\n[RANK {local_rank}] --> [AFTER PACKING] layer_numbers: {layer_numbers}')