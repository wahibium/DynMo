"""Transformer."""
import math
from contextlib import nullcontext
from mimetypes import init
import torch
import torch.nn.functional as F

from megatron import get_timers, get_args, get_global_memory_buffer
from megatron import mpu
from megatron.mpu.random import MakeViewlessTensor
from .module import MegatronModule
from megatron.model.enums import AttnMaskType, ModelType, LayerType, AttnType
from megatron.model import LayerNorm, SparseLinear, SparseCoreAttention
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.utils import attention_mask_func, openai_gelu, erf_gelu


""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""

class DropPath(MegatronModule):
    """Drop paths (Stochastic Depth) per sample 
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_state):
        if self.drop_prob == 0. or not self.training:
            return hidden_state
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (hidden_state.shape[0],) + (1,) * (hidden_state.ndim - 1)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=hidden_state.dtype, device=hidden_state.device)
        random_tensor.floor_()  # binarize
        output = hidden_state.div(keep_prob) * random_tensor
        return output

class SparseMLP(MegatronModule):
    """Sparse MLP.
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, init_method, output_layer_init_method):
        super(SparseMLP, self).__init__()
        self.args = get_args()
        self.to_4h = SparseLinear(self.args.hidden_size, self.args.ffn_hidden_size, init_method=init_method)
        self.to_h = SparseLinear(self.args.ffn_hidden_size, self.args.hidden_size, init_method=output_layer_init_method)
        self.dropout = torch.nn.Dropout(self.args.hidden_dropout)

    def forward(self, hidden_states):
        # PARAM = linear1.weight: (ffn_hidden_size, embedding_size) -> (2048, 512)
        # PARAM = linear2.weight: (embedding_size, ffn_hidden_size) -> (512, 2048)
        # IN = x: (batch_size, seq_len, embedding_size) -> (32, 72, 512)
        # OPERATION: x.(w.t()) -> (32, 72, 2048)
        # OPERATION: x.(w.t()) -> (32, 72, 512)
        # OUT = out: (batch_size, seq_len, embedding_size) -> (32, 72, 512)
        #print(f'[SparseMLP] --> hidden_states: {hidden_states.size()}')
        
        # [s,b,h]
        output_shape = hidden_states.size()
        
        # [b,s,h]
        hidden_states = hidden_states.permute(1, 0, 2).contiguous()
        
        # [b,s,4*h]
        #print(f'[to_4h] --> {hidden_states.size()}')
        first_linear_out = self.dropout(self.to_4h(hidden_states).transpose(1,2).contiguous() + self.to_4h.bias)
        
        # [b,s,h]
        #print(f'[to_h] --> {first_linear_out.size()}')
        output = self.to_h(first_linear_out).transpose(1,2).contiguous() + self.to_h.bias
        #output_bias = torch.zeros(self.args.hidden_size).cuda(int(self.args.rank))

        #print(f'[SparseMLP] --> output before reshape: {output.size()}')
        output = output.reshape(*output_shape)
        #print(f'[SparseMLP] --> {output}')
        #print(f'[SparseMLP] --> output after reshape: {output.size()}')

        return output, self.to_h.bias

class ParallelMLP(MegatronModule):
    """MLP.
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, init_method, output_layer_init_method):
        super(ParallelMLP, self).__init__()
        args = get_args()

        # Project to 4h.
        self.to_4h = mpu.ColumnParallelLinear(
            args.hidden_size,
            args.ffn_hidden_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True)

        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu
        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu

        # Project back to h.
        self.to_h = mpu.RowParallelLinear(
            args.ffn_hidden_size,
            args.hidden_size,
            bias=True,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    def forward(self, hidden_states):
        #print(f'[ParallelMLP] --> hidden_states: {hidden_states.size()}')

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.to_4h(hidden_states)

        if self.bias_gelu_fusion:
             intermediate_parallel = \
                     bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = \
                self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output, output_bias = self.to_h(intermediate_parallel)
        #print(f'[ParallelMLP] --> output: {output.size()}')

        return output, output_bias

class CoreAttention(MegatronModule):

    def __init__(self, layer_number, attn_mask_type=AttnMaskType.padding):
        super(CoreAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = args.sequence_parallel

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(projection_size, world_size)
        self.hidden_size_per_attention_head = mpu.divide(projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(args.num_attention_heads, world_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        #print(f'[CoreAttention] --> query_layer before view: {query_layer.size()}')

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = get_global_memory_buffer().get_tensor(
            (output_size[0]*output_size[1], output_size[2], output_size[3]),
            query_layer.dtype, "mpu")

        # Raw attention scores. [b * n, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))

        # change view to [b, n, sq, sk]
        attention_scores = matmul_result.view(*output_size)
        #print(f'[CoreAttention] --> scores: {attention_scores.size()}')

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        #print(f'scores: {attention_scores.size()}, mask: {attention_mask.size()}')
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)
        #print(f'[CoreAttention] --> weights: {attention_probs.size()}')

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.sequence_parallel:
            with mpu.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        #print(f'[CoreAttention] --> representations: {context_layer.size()}')

        return context_layer

class SparseAttention(MegatronModule):
    def __init__(self, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super().__init__()
        
        self.args = get_args()
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = self.args.params_dtype

        projection_size = self.args.kv_channels * self.args.num_attention_heads
        self.hidden_size_per_attention_head = mpu.divide(projection_size, self.args.num_attention_heads)
        self.num_attention_heads_per_partition = self.args.num_attention_heads

        self.query_key_value = SparseLinear(self.args.hidden_size, 3 * projection_size, init_method=init_method)

        self.dense = SparseLinear(projection_size, self.args.hidden_size, init_method=init_method)

        #self.core_attention = SparseCoreAttention(self.layer_number, self.attn_mask_type)
        self.core_attention = CoreAttention(self.layer_number, self.attn_mask_type)

    def forward(self, hidden_states, mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        # hidden_states: [s, b, h] --> [b, s, h]
        hidden_states = hidden_states.permute(1, 0, 2).contiguous()

        # mixed_x_layer: [b, s, (n * 3 * hn)]
        #print(f'[query_key_value] --> {hidden_states.size()}')
        mixed_x_layer = self.query_key_value(hidden_states).transpose(1, 2).contiguous() + self.query_key_value.bias
        #print(f'[Rank {torch.cuda.current_device()} SparseAttention] --> mixed_x_layer: {mixed_x_layer}')

        # new_tensor_shape: [b, s, n, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 3 * self.hidden_size_per_attention_head)
        
        # mixed_x_layer: [b, s, n, 3 * hn]
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape).contiguous()

        # query, key, value: each [b, s, n, hn]
        (query_layer, key_layer, value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)

        # context_layer: [s, b, h]
        #print(f'Layer Number: {self.layer_number}')
        query_layer = query_layer.permute(1,0,2,3).contiguous()
        key_layer = key_layer.permute(1,0,2,3).contiguous()
        value_layer = value_layer.permute(1,0,2,3).contiguous()
        #print(f'[Rank {torch.cuda.current_device()} SparseAttention] --> query_layer: {query_layer.size()}')
        context_layer = self.core_attention(query_layer, key_layer, value_layer, mask)

        #print(f'[RANK {torch.distributed.get_rank()}] Layer Number: {self.layer_number} --> dense values: {self.dense.values}')
        context_layer = context_layer.transpose(0, 1).contiguous()

        # output: [s, b, h]
        #print(f'[dense] --> {context_layer.size()}')
        output = self.dense(context_layer).transpose(1, 2).transpose(0, 1).contiguous() + self.dense.bias
        #bias = torch.zeros(self.args.hidden_size).cuda(int(self.args.rank))

        #print(f'[RANK {torch.distributed.get_rank()}]: {self.dense.bias}')

        return output, self.dense.bias

class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super(ParallelAttention, self).__init__()
        args = get_args()
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = args.params_dtype

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = mpu.divide(projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(args.num_attention_heads, world_size)

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                3 * projection_size,
                bias=True,
                gather_output=False,
                init_method=init_method)
        else:
            assert attention_type == AttnType.cross_attn
            self.query = mpu.ColumnParallelLinear(
                args.hidden_size,
                projection_size,
                bias=True,
                gather_output=False,
                init_method=init_method)

            self.key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                2 * projection_size,
                bias=True,
                gather_output=False,
                init_method=init_method)

        self.core_attention = CoreAttention(self.layer_number,
                                            self.attn_mask_type)
        self.checkpoint_core_attention = args.recompute_granularity == 'selective'

        # Output.
        self.dense = mpu.RowParallelLinear(
            projection_size,
            args.hidden_size,
            bias=True,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    def _checkpointed_attention_forward(self, query_layer, key_layer,
                                        value_layer, attention_mask):
        """Forward method with activation checkpointing."""
        def custom_forward(*inputs):
            query_layer = inputs[0]
            key_layer = inputs[1]
            value_layer = inputs[2]
            attention_mask = inputs[3]
            output_ = self.core_attention(query_layer, key_layer,
                                          value_layer, attention_mask)
            return output_

        hidden_states = mpu.checkpoint(
            custom_forward,
            False, query_layer, key_layer, value_layer, attention_mask)

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_len, batch_size):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=self.params_dtype,
            device=torch.cuda.current_device())

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, inference_params=None):
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 3 * self.hidden_size_per_attention_head)
                 
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer,
             key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)

        # ==================================
        # core attention computation
        # ==================================

        if self.checkpoint_core_attention:
            context_layer = self._checkpointed_attention_forward(
                query_layer, key_layer, value_layer, attention_mask)
        else:
            context_layer = self.core_attention(
                query_layer, key_layer, value_layer, attention_mask)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        #print(f'[ParallelAttention] --> Output: {output.size()}, bias: {bias.size()}')

        return output, bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x: torch.Tensor,
                                 bias: torch.Tensor,
                                 residual: torch.Tensor,
                                 prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x: torch.Tensor,
                                     bias: torch.Tensor,
                                     residual: torch.Tensor,
                                     prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method, output_layer_init_method,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0.):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm,
            sequence_parallel=args.sequence_parallel)

        if args.sparsify:
            self.self_attention = SparseAttention(
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.self_attn,
                attn_mask_type=self_attn_mask_type)
            # self.self_attention = ParallelAttention(
            #     init_method,
            #     output_layer_init_method,
            #     layer_number,
            #     attention_type=AttnType.self_attn,
            #     attn_mask_type=self_attn_mask_type)
        else:
            self.self_attention = ParallelAttention(
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.self_attn,
                attn_mask_type=self_attn_mask_type)

        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm,
            sequence_parallel=args.sequence_parallel)

        if self.layer_type == LayerType.decoder:
            self.inter_attention = ParallelAttention(
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon,
                no_persist_layer_norm=args.no_persist_layer_norm,
                sequence_parallel=args.sequence_parallel)

        # MLP
        if args.sparsify:
            self.mlp = SparseMLP(init_method, output_layer_init_method)
        else:
            self.mlp = ParallelMLP(init_method, output_layer_init_method)

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = \
                nullcontext if use_nvfuser else torch.enable_grad

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, attention_bias = \
            self.self_attention(
                layernorm_output,
                attention_mask,
                inference_params=inference_params)

        #print(f'[Rank {torch.cuda.current_device()} Self Attention] --> output: {attention_output}')

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        if self.drop_path is None:
            # jit scripting for a nn.module (with dropout) is not
            # trigerring the fusion kernel. For now, we use two
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.
            if self.bias_dropout_fusion:
                if self.training:
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                bias_dropout_add_func = get_bias_dropout_add(self.training)

            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
        else:
            out = torch.nn.functional.dropout(attention_output + attention_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
            layernorm_input = residual + self.drop_path(out)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        #print(f'[Rank {torch.cuda.current_device()} Layer Norm] --> output: {layernorm_output}')

        if self.layer_type == LayerType.decoder:
            attention_output, attention_bias = \
                self.inter_attention(layernorm_output,
                                     enc_dec_attn_mask,
                                     encoder_output=encoder_output)
            # residual connection
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = layernorm_input

            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)

            # Layer norm post the decoder attention
            layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)
        #print(f'[Rank {torch.cuda.current_device()} MLP] --> output: {mlp_output}')

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        if self.drop_path is None:
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(
                    mlp_output,
                    mlp_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)

            # Jit compiled function creates 'view' tensor. This tensor
            # potentially gets saved in the MPU checkpoint function context,
            # which rejects view tensors. While making a viewless tensor here
            # won't result in memory savings (like the data loader, or
            # p2p_communication), it serves to document the origin of this
            # 'view' tensor.
            output = mpu.make_viewless_tensor(inp = output,
                                              requires_grad = output.requires_grad,
                                              keep_graph = True)

        else:
            out = torch.nn.functional.dropout(mlp_output + mlp_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
            output = residual + self.drop_path(out)

        #print(f'[Rank {torch.cuda.current_device()} TransformerLayer] --> output: {output}')

        return output


class NoopTransformerLayer(MegatronModule):
    """A single 'no-op' transformer layer.

    The sole purpose of this layer is for when a standalone embedding layer
    is used (i.e., args.standalone_embedding_stage == True). In this case,
    zero transformer layers are assigned when pipeline rank == 0. Additionally,
    when virtual pipeline rank >= 1, zero total model parameters are created
    (virtual rank 0 contains the input embedding). This results in the model's
    input and output tensors being the same, which causes an error when
    performing certain memory optimiations on the output tensor (e.g.,
    deallocating it). Thus, this layer disconnects the input from the output
    via a clone. Since ranks containing a no-op layer are generally under-
    utilized (both compute and memory), there's no worry of any performance
    degredation.
    """

    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        return hidden_states.clone()


class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, init_method, output_layer_init_method,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 post_layer_norm=True, 
                 pre_process=True, post_process=True,
                 drop_path_rate=0.0):
        super(ParallelTransformer, self).__init__()
        args = get_args()

        self.layer_type = layer_type
        self.model_type = args.model_type
        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.drop_path_rate = drop_path_rate

        # Store activation checkpoiting flag.
        self.recompute_granularity = args.recompute_granularity
        self.recompute_method = args.recompute_method
        self.recompute_num_layers = args.recompute_num_layers
        self.distribute_saved_activations = \
            args.distribute_saved_activations and not args.sequence_parallel

        self.sequence_parallel = args.sequence_parallel

        # Number of layers.
        self.num_layers = mpu.get_num_layers(
            args, args.model_type == ModelType.encoder_and_decoder)

        self.drop_path_rates = [rate.item() for rate in torch.linspace(0, self.drop_path_rate, args.num_layers)]

        # Transformer layers.
        def build_layer(layer_number):
            return ParallelTransformerLayer(
                init_method,
                output_layer_init_method,
                layer_number,
                layer_type=layer_type,
                self_attn_mask_type=self_attn_mask_type,
                drop_path_rate=self.drop_path_rates[layer_number - 1])
        
        if args.virtual_pipeline_model_parallel_size is not None:
            assert args.num_layers % args.virtual_pipeline_model_parallel_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
            assert args.model_type != ModelType.encoder_and_decoder
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // args.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
                args.num_layers // args.virtual_pipeline_model_parallel_size) + \
                (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            if args.model_type == ModelType.encoder_and_decoder and \
                    mpu.get_pipeline_model_parallel_world_size() > 1:
                pipeline_rank = mpu.get_pipeline_model_parallel_rank()
                if layer_type == LayerType.encoder:
                    offset = pipeline_rank * self.num_layers
                else:
                    num_ranks_in_enc = args.pipeline_model_parallel_split_rank
                    offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
            else:
                offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers

        if self.num_layers == 0:
            # When a standalone embedding stage is used (e.g.,
            # args.standalone_embedding_stage == True), virtual pipeline ranks
            # on pipeline rank 0 will have zero transformer layers assigned to
            # them. This results in the model's input and output tensors to be
            # the same, which will cause failure for certain output tensor
            # optimizations (e.g., pipeline output deallocation). To remedy
            # this, we assign a 'no-op' layer on these ranks, which will
            # disconnect the input tensor from the output tensor.
            self.num_layers = 1
            self.layers = torch.nn.ModuleList([ NoopTransformerLayer(1) ])
        else:
            self.layers = torch.nn.ModuleList(
                [build_layer(i + 1 + offset) for i in range(self.num_layers)])

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon,
                no_persist_layer_norm=args.no_persist_layer_norm,
                sequence_parallel=args.sequence_parallel)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask,
                              encoder_output, enc_dec_attn_mask):
        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                attention_mask = inputs[1]
                encoder_output = inputs[2]
                enc_dec_attn_mask = inputs[3]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, attention_mask, encoder_output, enc_dec_attn_mask)
                return x_
            return custom_forward

        if self.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            l = 0
            while l < self.num_layers:
                hidden_states = mpu.checkpoint(
                    custom(l, l + self.recompute_num_layers),
                    self.distribute_saved_activations,
                    hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
                l += self.recompute_num_layers

        elif self.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for l in range(self.num_layers):
                if l < self.recompute_num_layers:
                    hidden_states = mpu.checkpoint(
                        custom(l, l + 1),
                        self.distribute_saved_activations,
                        hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
                else:
                    hidden_states = custom(l, l + 1)(
                        hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        if len(self.layers) == 0:
            if not self.pre_process:
                hidden_states = self.input_tensor
            #print(f'[RANK {torch.distributed.get_rank()}] --> Hidden states: {hidden_states}')
            hidden_states = MakeViewlessTensor.apply(hidden_states, True)
            return hidden_states
        # hidden_states: [s, b, h]
        ### Mine ###
        args = get_args()
        timers = get_timers()
        ###

        # Checks.
        if inference_params:
            assert self.recompute_granularity is None, \
                'inference does not work with activation checkpointing'

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        # 
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = mpu.make_viewless_tensor(
            hidden_states,
            requires_grad=True,
            keep_graph=True,
        )

        if self.sequence_parallel:
            rng_context = mpu.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        with rng_context:
            # Forward pass.
            if self.recompute_granularity == 'full':
                hidden_states = self._checkpointed_forward(hidden_states,
                                                           attention_mask,
                                                           encoder_output,
                                                           enc_dec_attn_mask)
            else:
                for layer in self.layers:
                    if args.profile:
                        timers(f'layer_{layer.layer_number - 1}').start()
                    
                    #print(f'[RANK {torch.distributed.get_rank()}] --> Executing layer_number: {layer.layer_number - 1}')
                    hidden_states = layer(
                        hidden_states,
                        attention_mask,
                        encoder_output=encoder_output,
                        enc_dec_attn_mask=enc_dec_attn_mask,
                        inference_params=inference_params)

                    if args.profile:
                        timers(f'layer_{layer.layer_number - 1}').stop()

        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states
