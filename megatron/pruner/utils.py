import torch

def get_pruning_levels(args):
    """ Pruning schedule from: https://arxiv.org/abs/1710.01878 """
    sparsities = []
    s_initial = 0   # initial sparsity
    s_final = args.final_sparsity   # final sparsity
    t0 = args.pruning_start_iter         # starting step
    t_end = args.pruning_end_iter       # end step
    delta_t = args.pruning_freq    # pruning frequency
    n = (t_end - t0) // delta_t           # pruning steps

    for t in range(t0, t_end):
        if t % delta_t == 0:
            st = s_final + (s_initial - s_final) * ((1 - (t - t0) / (n * delta_t)) ** 3)
            sparsities.append(st)
            
    sparsities.insert(0, 0)
    percentages = []
    for idx in range(1,len(sparsities)):
        percentages.append((sparsities[idx] - sparsities[idx - 1]) / (1 - sparsities[idx - 1]))

    print(f'Pruning Levels: {sparsities}, Percentage in each step: {percentages}')
    return percentages

def disable_sparse_grad(module):
    for name, param in module.named_parameters():
        if 'values' in name:
            param.requires_grad = False

def disable_grad(module):
    for name, param in module.named_parameters():
        if 'weight' in name and 'norm' not in name and 'embedding' not in name and 'pooler' not in name and 'lm_head' not in name and 'binary' not in name:
            param.requires_grad = False

def enable_sparse_grad(module):
    for name, param in module.named_parameters():
        if 'values' in name:
            param.requires_grad = True

def enable_grad(module):
    for name, param in module.named_parameters():
        if 'weight' in name and 'norm' not in name and 'embedding' not in name and 'pooler' not in name and 'lm_head' not in name and 'binary' not in name:
            param.requires_grad = True

def concat_sparse_params(module):
    params = [param for name, param in module.named_parameters() if 'values' in name]
    return params

def concat_column_indices(module):
    return torch.cat([buffer for name, buffer in module.named_buffers() if 'column_indices' in name])

def concat_row_offsets(model):
    return [buffer for name, buffer in model.named_buffers() if 'row_offsets' in name]

def concat_dense_params(module):
    params = []

    for name, param in module.named_parameters():
        if 'weight' in name and \
           'norm' not in name and \
           'embedding' not in name and \
           'pooler' not in name and \
           'lm_head' not in name and \
           'binary' not in name:
            
            params.append(param.flatten())

    params = torch.cat(params)

    return params

def get_param_range_dict(module):
    keys = []
    shapes_and_ranges = []
    param_range_start = 0
    for name, param in module.named_parameters():
        if 'weight' in name and 'norm' not in name and 'embedding' not in name and 'pooler' not in name and 'lm_head' not in name and 'binary' not in name:
            keys.append(name[7:])

            param_range_end = param_range_start + param.numel()
            shapes_and_ranges.append((tuple(param.shape), (param_range_start, param_range_end)))

            param_range_start = param_range_end

    param_range_dict = dict(zip(keys, shapes_and_ranges))
    
    return param_range_dict

def get_sparse_param_sizes(module):
    keys = []
    sizes = []
    for name, param in module.named_parameters():
        if 'values' in name:
            keys.append(name[7:])
            sizes.append(param.size(0))

    param_size_dict = dict(zip(keys, sizes))
    
    return param_size_dict
