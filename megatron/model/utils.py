"""Utilities for models."""
import math

import torch

from megatron import get_args

def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


def get_linear_layer(rows, columns, init_method):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    if get_args().perform_initialization:
        init_method(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
    return layer

@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))
def openai_gelu(x):
    return gelu_impl(x)

#This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype)+torch.ones_like(x).to(dtype=x.dtype))

def dense_to_sparse_3d(mask):
    values_list = []
    row_indices_list = []
    row_offsets_list = []
    column_indices_list = []
    nnzs = []

    for index in range(mask.size(0)):
        values, row_indices, row_offsets, column_indices, nnz = dense_to_sparse(mask[index, :, :])
        values_list.append(values)
        row_indices_list.append(row_indices)
        row_offsets_list.append(row_offsets)
        column_indices_list.append(column_indices)
        nnzs.append(nnz)

    values = torch.cat(values_list)
    row_indices = torch.stack(row_indices_list)
    row_offsets = torch.stack(row_offsets_list)
    column_indices = torch.cat(column_indices_list)
    nnzs = torch.tensor(nnzs).cuda(torch.cuda.current_device())

    return values, row_indices, row_offsets, column_indices, nnzs

def dense_to_sparse(matrix):
    csr = matrix.to_sparse_csr()
    values = csr.values().detach().to(torch.float32).requires_grad_(True)
    row_offsets = csr.crow_indices().to(torch.int32)
    row_indices = diffsort(row_offsets)
    column_indices = csr.col_indices().to(torch.int32)

    return values, row_indices, row_offsets, column_indices, csr._nnz()

def diffsort(offsets):
    #print(f'[DIFFSORT] --> offsets size: {offsets.size()}, offsets: {offsets}')
    diffs = (offsets - torch.roll(offsets, -1, 0))[:-1]
    #print(f'[DIFFSORT] --> diffs: {diffs.size()}')
    return torch.argsort(diffs, descending=True).to(torch.int32)

def diffsort_many_mask(offsets):
    num_masks = offsets.size(0)

    row_indices_list = []

    for idx in range(num_masks):
        indices = diffsort(offsets[idx])
        row_indices_list.append(indices)

    return torch.stack(row_indices_list)
