import time
import numpy as np
import math
import torch
import torch.nn as nn
import torch_sputnik

from megatron import mpu
from .module import MegatronModule
from megatron import get_args
from megatron.model.enums import AttnMaskType
from megatron.model.utils import *

class SparseLinear(MegatronModule):

    def __init__(self, input_features, output_features, init_method, bias=False):
        super(SparseLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.init_method = init_method

        self.init_weights()

    def init_weights(self):
        weight = torch.empty(self.output_features, self.input_features)
        self.init_method(weight)
        weight[weight == 0] = np.random.normal(loc=0.08, scale=0.02)
        self.weight = nn.Parameter(weight)

        self.bias = nn.Parameter(torch.zeros(self.output_features, device=torch.cuda.current_device(), dtype=torch.float32))

        self.setup_sparse_tensors()
        
    def setup_sparse_tensors(self):
        values, row_indices, row_offsets, column_indices, _ = dense_to_sparse(self.weight)
        self.values = nn.Parameter(values)
        self.register_buffer("row_indices", row_indices)
        self.register_buffer("row_offsets", row_offsets)
        self.register_buffer("column_indices", column_indices)

        del self.weight
        #del self.bias
        
    def forward(self, x):
        #print(self.output_features, self.input_features, self.values.size()[0], self.row_indices.size()[0], self.row_offsets.size()[0], self.column_indices.size()[0])
        #print(self.values.device, self.row_indices.device, self.row_offsets.device, self.column_indices.device)
        return SparseLinearFunction.apply(self.output_features, self.input_features, self.values, self.row_indices, self.row_offsets, self.column_indices, x.transpose(1,2).contiguous(), self.bias)

class SparseCoreAttention(MegatronModule):
    def __init__(self, layer_number, attn_mask_type=AttnMaskType.padding):
        super(SparseCoreAttention, self).__init__()
        self.args = get_args()

        projection_size = self.args.kv_channels * self.args.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = mpu.divide(projection_size, self.args.num_attention_heads)
        self.num_attention_heads_per_partition = self.args.num_attention_heads

        self.sddmm = Sddmm.apply
        self.softmax = CsrSoftmax.apply
        self.spmm = Spmm.apply

    def four_d_to_three_d(self, tensor):
        b, n, s, hn = tensor.size()

        return tensor.reshape(b * n, s, hn)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        batch_size = query_layer.size(0)
        values, row_indices, row_offsets, column_indices, nnzs = dense_to_sparse_3d(attention_mask)

        # output_shape: [s, b, h]
        output_shape = (query_layer.size(1),
                       query_layer.size(0),
                       query_layer.size(2) * query_layer.size(3))

        #print(f'[SparseCoreAttention] --> query_layer before permute: {query_layer.size()}')

        # Input query_layer, key_layer, value_layer: each [b, s, n, hn] --> [b, n, s, hn]
        query_layer = torch.permute(query_layer, (0, 2, 1, 3))
        key_layer = torch.permute(key_layer, (0, 2, 1, 3))
        value_layer = torch.permute(value_layer, (0, 2, 1, 3))

        #print(f'[Rank {torch.cuda.current_device()} SparseCoreAttention] --> query_layer after permute: {query_layer}')

        # Query_layer, key_layer, value_layer: each [b, n, s, hn] --> [b * n, s, hn]
        query_layer = self.four_d_to_three_d(query_layer)
        key_layer = self.four_d_to_three_d(key_layer)
        value_layer = self.four_d_to_three_d(value_layer)

        #print(f'[Rank {torch.cuda.current_device()} SparseCoreAttention] --> query_layer 3d: {query_layer}')
        scores = self.sddmm(batch_size,
                    self.args.seq_length, self.args.seq_length,
                    nnzs,
                    row_indices, 
                    row_offsets, 
                    column_indices, 
                    query_layer, 
                    key_layer
        ) / math.sqrt(self.hidden_size_per_attention_head)

        #print(f'[Rank {torch.cuda.current_device()} SparseCoreAttention] --> scores: {scores.size()}, column_indices: {column_indices.size()}')

        weights = self.softmax(
                    batch_size, self.args.seq_length, nnzs,
                    scores, 
                    row_indices, 
                    row_offsets, 
                    column_indices
        )

        #print(f'[Rank {torch.cuda.current_device()} SparseCoreAttention] --> weights: {weights}')
        
        # [b * n, s, hn]
        representations = self.spmm(batch_size,
                self.args.seq_length, self.args.seq_length,
                nnzs,
                weights,
                row_indices, 
                row_offsets, 
                column_indices, 
                value_layer
        )

        #print(f'query: {query_layer.size()}, scores: {scores.size()}, weights: {weights.size()}, mask: {attention_mask.size()}')

        #print(f'[Rank {torch.cuda.current_device()} SparseCoreAttention] --> representations before reshape: {representations}')
        representations = torch.permute(representations, (1, 0, 2)).reshape(*output_shape)
        #print(f'[Rank {torch.cuda.current_device()} SparseCoreAttention] --> representations after reshape: {representations}')

        #print(f'[SparseCoreAttention] --> representations: {representations}')

        # Output: [s, b, h]
        return representations

class SparseLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, m, k, values, row_indices, row_offsets, column_indices, dense, bias):
        ctx.m = m
        ctx.k = k
        ctx.row_indices = row_indices
        ctx.row_offsets = row_offsets
        ctx.column_indices = column_indices
        ctx.save_for_backward(values, dense)

        #print(f'[RANK {torch.distributed.get_rank()} (Forward)] --> m: {m}, k: {k},' +
        #      f'values: {values.size()}, row_indices: {row_indices.size()}, row_offsets: {row_offsets.size()}' + 
        #      f', column_indices: {column_indices.size()}, dense: {dense.size()}')
        result = torch_sputnik.left_spmm(m, k, values, row_indices, row_offsets, column_indices, dense)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        #print(f'Backward started')
        m = ctx.m
        k = ctx.k
        row_indices = ctx.row_indices
        row_offsets = ctx.row_offsets
        column_indices = ctx.column_indices
        values, dense = ctx.saved_tensors

        grad_m = grad_k = grad_values = grad_row_indices = grad_row_offsets = grad_column_indices = grad_dense = grad_bias = None

        # sparse matrix grad
        #print(f'Backward sddmm')
        #print(f'[RANK {torch.distributed.get_rank()}] --> m:{m}, k:{k}, row_indices: {row_indices.size()}, row_offsets: {row_offsets.size()}, column_indices: {column_indices.size()}, grad_output: {grad_output.size()}, dense: {dense.size()}')
        grad_values = torch_sputnik.sddmm(m, k,
                                        row_indices, 
                                        row_offsets, 
                                        column_indices,
                                        grad_output, 
                                        dense)
        #print(f'[RANK {torch.distributed.get_rank()}] --> m:{m}, k:{k}, values: {values.size()}, row_offsets: {row_offsets.size()}, column_indices: {column_indices.size()}')
        #print(f'[RANK {torch.distributed.get_rank()}] --> m:{m}, k:{k}, values: {values}, row_offsets: {row_offsets}, column_indices: {column_indices}')
        values_t, row_offsets_t, column_indices_t = torch_sputnik.csr_transpose(m, k, 
                                                                                values, 
                                                                                row_offsets, 
                                                                                column_indices)
       
        row_indices_t = diffsort(row_offsets_t)

        # dense matrix grad
        #print(f'[RANK {torch.distributed.get_rank()}] --> Before backward left spmm, values_t: {values_t.size()}')
        grad_dense = torch_sputnik.left_spmm(k, m, 
                                        values_t, 
                                        row_indices_t, 
                                        row_offsets_t, 
                                        column_indices_t, 
                                        grad_output)
        
        grad_bias = grad_output.sum(dim=[0,2])
        #print("Backward finished")
        #print(f'Grad values: {grad_values}')
        return grad_m, grad_k, grad_values, grad_row_indices, grad_row_offsets, grad_column_indices, grad_dense, grad_bias

class Spmm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, b, m, k, nonzeros, values, row_indices, row_offsets, column_indices, dense):
        ctx.b = b
        ctx.m = m
        ctx.k = k
        ctx.nonzeros = nonzeros
        ctx.row_indices = row_indices
        ctx.row_offsets = row_offsets
        ctx.column_indices = column_indices
        ctx.save_for_backward(values, dense)

        #print(f'm: {m}, k: {k}, values: {values.size()}, row_indices: {row_indices.size()}, row_offsets: {row_offsets.size()}, column_indices: {column_indices.size()}, dense: {dense.size()}')
        
        result = torch_sputnik.spmm_many_mask(b, m, k, nonzeros, values, row_indices, row_offsets, column_indices, dense)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        #print(f'Spmm backward works')
        b = ctx.b
        m = ctx.m
        k = ctx.k
        nonzeros = ctx.nonzeros
        row_indices = ctx.row_indices
        row_offsets = ctx.row_offsets
        column_indices = ctx.column_indices
        values, dense = ctx.saved_tensors

        grad_b = grad_m = grad_k = grad_nonzeros = grad_values = grad_row_indices = grad_row_offsets = grad_column_indices = grad_dense = None

        # sparse matrix grad
        #print(f'[SpMM GRAD] grad_output: {grad_output.size()}, dense: {dense.size()}')
        # grad_output: [32, 512, 64], row_indices: [2048], row_offsets: [2052], column_indices: [525312], dense: [32, 512, 64]
        grad_values = torch_sputnik.sddmm_many_mask(b, m, k, nonzeros,
                                        row_indices, 
                                        row_offsets, 
                                        column_indices,
                                        grad_output, 
                                        dense)

        #print(f'[SpMM GRAD] csr_transpose --> m: {m}, k: {k}, values: {values.size()}, row_offsets: {row_offsets.size()}, column_indices: {column_indices.size()}')
        # m: 512, k: 512, values: [32, 131328]
        #print(f'SpMM backward csr transpose: row_offsets: {row_offsets}')
        values_t, row_offsets_t, column_indices_t = torch_sputnik.csr_transpose_many_mask(b, m, k, nonzeros, 
                                    values,
                                    row_offsets,
                                    column_indices)
        #print(f'SpMM backward csr transpose: row_offsets_t: {row_offsets_t}')
        row_indices_t = diffsort_many_mask(row_offsets_t)

        # dense matrix grad
        #print(f'[SpMM GRAD] spmm --> k: {k}, m: {k}, nonzeros: {nonzeros}, values_t: {values_t.size()}, row_indices_t: {row_indices_t.size()}, row_offsets_t: {row_offsets_t.size()}, column_indices_t: {column_indices_t.size()}, grad_output: {grad_output.size()}')
        grad_dense = torch_sputnik.spmm_many_mask(b, k, m, nonzeros, 
                                        values_t, 
                                        row_indices_t, 
                                        row_offsets_t, 
                                        column_indices_t, 
                                        grad_output)

        #print(f'Spmm backward finished')

        return grad_b, grad_m, grad_k, grad_nonzeros, grad_values, grad_row_indices, grad_row_offsets, grad_column_indices, grad_dense

class CsrSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b, m, nonzeros, scores, row_indices, row_offsets, column_indices):
        ctx.b = b
        ctx.m = m
        ctx.nonzeros = nonzeros
        ctx.scores = scores
        ctx.row_indices = row_indices
        ctx.row_offsets = row_offsets
        ctx.column_indices = column_indices

        output = torch_sputnik.sparse_softmax_many_mask(
                    b, m, nonzeros,
                    scores, 
                    row_indices, 
                    row_offsets, 
                    column_indices)
        
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        #print(f'Softmax backward works')
        b = ctx.b
        m = ctx.m
        nonzeros = ctx.nonzeros
        scores = ctx.scores
        row_indices = ctx.row_indices
        row_offsets = ctx.row_offsets
        column_indices = ctx.column_indices
        output = ctx.saved_tensors
        output = output[0]

        grad_b = grad_m = grad_nonzeros = grad_scores = grad_row_indices = grad_row_offsets = grad_column_indices = None

        #I = torch.eye(grad_output.shape[0], grad_output.shape[1]).cuda(torch.cuda.current_device())

        #grad_scores = torch.empty_like(output)
        #for batch_idx in range(grad_output.size(-1)):
        #    softmax = output[batch_idx].view(1, -1)
        #    grad    = grad_output[batch_idx].view(1, -1)
        #    d_softmax = softmax * torch.eye(softmax.size(0)).cuda(torch.cuda.current_device()) - (softmax.t() @ softmax)
        #    grad_scores[batch_idx] = grad @ d_softmax

        #softmax = torch.nn.functional.softmax(grad_output, dim=1)
        #softmax = torch_sputnik.sparse_softmax_many_mask(
        #                b, m, nonzeros,
        #                grad_output, 
        #                row_indices, 
        #                row_offsets, 
        #                column_indices)
        #print(f'output: {output.size()}, grad_output: {grad_output.size()}')
        grad_scores = (-output) * grad_output
        #print(f'grad_scores: {grad_scores}, output: {output}, grad_output: {grad_output}')

        #print(f'Softmax backward finished')

        return grad_b, grad_m, grad_nonzeros, grad_scores, grad_row_indices, grad_row_offsets, grad_column_indices

class Sddmm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, b, m, n, nonzeros, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix):
        ctx.b = b
        ctx.m = m
        ctx.n = n
        ctx.nonzeros = nonzeros
        ctx.row_indices = row_indices
        ctx.row_offsets = row_offsets
        ctx.column_indices = column_indices
        ctx.save_for_backward(lhs_matrix, rhs_matrix)

        #print(f'b: {b}, m: {m}, n: {n}, nonzeros: {nonzeros}, row_indices: {row_indices.size()}, row_offsets: {row_offsets.size()}, column_indices: {column_indices.size()}, lhs_matrix: {lhs_matrix.size()}')
        result = torch_sputnik.sddmm_many_mask(b, m, n, nonzeros, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        #print(f'Sddmm backward works')
        b = ctx.b
        m = ctx.m
        n = ctx.n
        nonzeros = ctx.nonzeros
        row_indices = ctx.row_indices
        row_offsets = ctx.row_offsets
        column_indices = ctx.column_indices
        lhs_matrix, rhs_matrix = ctx.saved_tensors

        grad_b = grad_m = grad_n = grad_nonzeros = grad_row_indices = grad_row_offsets = grad_column_indices = grad_lhs = grad_rhs = None
        
        # lhs grad
        #print(f'[SDDMM GRAD] b: {b}, m: {m}, n: {n}, nonzeros: {nonzeros}, grad_output: {grad_output.size()}, row_offsets: {row_offsets.size()}, column_indices: {column_indices.size()}, rhs_matrix: {rhs_matrix.size()}')
        # b: 4, m: 512, n: 512, nonzeros: [4], grad_output: [32, 131328], row_offsets: [2052], column_indices: [525312], rhs_matrix: [32, 512, 64]
        grad_lhs = torch_sputnik.spmm_many_mask(b, m, n, nonzeros, 
                                    grad_output,
                                    row_indices, 
                                    row_offsets, 
                                    column_indices, 
                                    rhs_matrix)

        #print(f'[SDDMM GRAD] grad_output: {grad_output.size()}, row_offsets: {row_offsets.size()}, column_indices: {column_indices.size()}')
        # grad_output: [32, 131328], row_offsets: [2052], column_indices: [525312]

        #print(f'SDDMM backward csr transpose: row_offsets: {row_offsets}')
        grad_t, row_offsets_t, column_indices_t = torch_sputnik.csr_transpose_many_mask(b, m, n, nonzeros, 
                                    grad_output, 
                                    row_offsets, 
                                    column_indices)

        #print(f'[SDDMM GRAD] grad_t: {grad_t.size()}, row_offsets_t: {row_offsets_t.size()}, column_indices_t: {column_indices_t.size()}')
        # grad_output_t: [32, 131328], row_offsets_t: [2052], column_indices_t: [525312]
        #print(f'SDDMM backward csr transpose: row_offsets_t: {row_offsets_t}')
        row_indices_t = diffsort_many_mask(row_offsets_t)

        # rhs grad
        #print(f'[SDDMM GRAD] grad_t: {grad_t.size()}, lhs_matrix: {lhs_matrix.size()}')
        # grad_output_t: [32, 131328], row_offsets_t: [2052], column_indices_t: [525312]
        grad_rhs = torch_sputnik.spmm_many_mask(b, n, m, nonzeros,
                                    grad_t, 
                                    row_indices_t, 
                                    row_offsets_t, 
                                    column_indices_t, 
                                    lhs_matrix)

        #print(f'Sddmm backward finished')

        return grad_b, grad_m, grad_n, grad_nonzeros, grad_row_indices, grad_row_offsets, grad_column_indices, grad_lhs, grad_rhs
