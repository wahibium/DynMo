import torch
import torch.distributed as dist
from dataclasses import dataclass
from megatron.model.enums import AttnMaskType, LayerType
from megatron.model.transformer import ParallelTransformerLayer
from megatron import get_values, get_column_indices, get_row_offsets
from megatron.model.utils import diffsort, init_method_normal, scaled_init_method_normal

@dataclass
class Transfer:
    src: int
    dst: int
    layer_idx: int

def create_layer(args, pair_rank, local_rank):
    drop_path_rates = [rate.item() for rate in torch.linspace(0, 0.0, args.num_layers)]
    init_method = init_method_normal(args.init_method_std)
    output_layer_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)

    # 0. Receive layer number
    layer_number = torch.cuda.LongTensor([-1])
    dist.recv(tensor=layer_number, src=pair_rank)

    # 1. Create a new transformer layer
    transformer_layer = ParallelTransformerLayer(
        init_method,
        output_layer_init_method,
        layer_number.item(),
        layer_type=LayerType.encoder,
        self_attn_mask_type=AttnMaskType.padding,
        drop_path_rate=drop_path_rates[layer_number.item() - 1]).cuda(local_rank)

    return transformer_layer

def send_layer_data(transformer_layer, pair_rank):
    # 0. Send layer number
    dist.send(tensor=torch.cuda.LongTensor([transformer_layer.layer_number]), dst=pair_rank)

    # 1. Send values
    values_sizes, values = get_values(transformer_layer)
    dist.send(tensor=torch.cuda.LongTensor(values_sizes), dst=pair_rank)
    dist.send(tensor=values, dst=pair_rank)

    # 2. Send column indices
    _, column_indices = get_column_indices(transformer_layer)
    dist.send(tensor=column_indices, dst=pair_rank)

    # 3. Send row offsets
    offset_sizes, row_offsets = get_row_offsets(transformer_layer)
    dist.send(tensor=torch.cuda.LongTensor(offset_sizes), dst=pair_rank)
    dist.send(tensor=row_offsets, dst=pair_rank)

def recv_layer_data(transformer_layer, pair_rank, local_rank):
    # 2. Receive values
    values_sizes = torch.cuda.LongTensor([0 for name, _ in transformer_layer.named_parameters() if 'values' in name])
    dist.recv(tensor=values_sizes, src=pair_rank)

    values = torch.cat([torch.zeros(size.item()).cuda(local_rank) for size in values_sizes])
    dist.recv(tensor=values, src=pair_rank)

    # 3. Receive column_indices
    column_indices_sizes = values_sizes.clone()
    column_indices = torch.cat([torch.zeros(size.item()).cuda(local_rank).to(torch.int32) for size in column_indices_sizes])
    dist.recv(tensor=column_indices, src=pair_rank)

    # 4. Receive row_offsets
    row_offsets_sizes = torch.cuda.LongTensor([0 for name, _ in transformer_layer.named_parameters() if 'values' in name])
    dist.recv(tensor=row_offsets_sizes, src=pair_rank)

    row_offsets = torch.cat([torch.zeros(size.item()).cuda(local_rank).to(torch.int32) for size in row_offsets_sizes])
    dist.recv(tensor=row_offsets, src=pair_rank)

    # 5. Calculate row indices based on row offsets
    all_indices = []
    offset_start = 0
    for size in row_offsets_sizes:
        offsets = row_offsets[offset_start: offset_start + size.item()]
        all_indices.append(diffsort(offsets))
    
    row_indices = torch.cat(all_indices)

    return values_sizes, values, column_indices, row_offsets, row_indices

def perform_transfers(model, transfers, local_rank, args, balancer):
    # Send and receive the parameters
    if len(transfers) > 0:
        for layer_idx, pair_rank, comm_type in transfers:
            if comm_type == 0:
                transformer_layer = model.module.language_model.encoder.layers[layer_idx]

                send_layer_data(transformer_layer, pair_rank)

                # 4. remove the sent layer
                if "diffusion" in balancer:
                    del model.module.language_model.encoder.layers[layer_idx]
                    model.module.language_model.encoder.num_layers -= 1
            else:
                transformer_layer = create_layer(args, pair_rank, local_rank)

                values_sizes, values, column_indices, row_offsets, row_indices = recv_layer_data(transformer_layer, pair_rank, local_rank)

                values_start = 0
                row_offsets_start = 0
                row_indices_start = 0
                values_idx = 0
                for name, _ in transformer_layer.named_parameters():
                    extension = name.split('.')[-1]
                    if extension == 'values':
                        size = values_sizes[values_idx]
                        values_idx += 1
                        module_name = name[:-7]
                    
                        submodule = transformer_layer.get_submodule(module_name)
                        m = submodule.output_features

                        csr_values = torch.nn.Parameter(values[values_start: values_start + size.item()])
                        csr_column_indices = column_indices[values_start: values_start + size.item()]
                        csr_row_offsets = row_offsets[row_offsets_start: row_offsets_start + m + 1]
                        csr_row_indices = row_indices[row_indices_start: row_indices_start + m]

                        submodule.values = csr_values
                        submodule.column_indices = csr_column_indices
                        submodule.row_offsets = csr_row_offsets
                        submodule.row_indices = csr_row_indices
                    
                        values_start += size.item()
                        row_offsets_start += (m + 1)
                        row_indices_start += m

                # 6. Add received layer to the encoder layer module list
                model.module.language_model.encoder.layers.append(transformer_layer)
                model.module.language_model.encoder.num_layers += 1

        if "deepspeed" in balancer:
            transfers.sort(reverse=True)
            for layer_idx, _, comm_type in transfers:
                if comm_type == 0:
                    # Remove the sent layer
                    del model.module.language_model.encoder.layers[layer_idx]
                    model.module.language_model.encoder.num_layers -= 1