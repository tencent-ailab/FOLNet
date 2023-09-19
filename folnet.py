import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import math
import copy
import numpy as np
from torch.nn import CrossEntropyLoss
from scipy.linalg import toeplitz
from itertools import permutations
from config_folnet import FOLNetConfig


def _gelu_python(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

gelu = _gelu_python if torch.__version__ < "1.4.0" else F.gelu

ACT2FN = {
    "gelu": gelu,
    "relu": F.relu,
}


try:
    import apex
    import apex.normalization
    from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction
    from apex.normalization.fused_layer_norm import FusedLayerNormFunction
    APEX_IS_AVAILABLE = True and torch.cuda.is_available()
except ImportError:
    print("Better speed can be achieved with Nvidia apex package")
    APEX_IS_AVAILABLE = False
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, elementwise_affine=False):
        super(LayerNorm, self).__init__()
        self.elementwise_affine = elementwise_affine
        self.shape = torch.Size((hidden_size,))
        self.eps = eps
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight = None
            self.bias = None
        self.apex_enabled = APEX_IS_AVAILABLE

    @torch.jit.unused
    def fused_layer_norm(self, x):
        if self.elementwise_affine:
            return FusedLayerNormAffineFunction.apply(
                x, self.weight, self.bias, self.shape, self.eps
            )
        else:
            return FusedLayerNormFunction.apply(x, self.shape, self.eps)

    def forward(self, x):
        if self.apex_enabled and not torch.jit.is_scripting():
            x = self.fused_layer_norm(x)
        else:
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if self.elementwise_affine:
                x = self.weight * x + self.bias
        return x


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        for k, v in config.__dict__.items():
            setattr(self, k, v)
        if not hasattr(config, "absolute_position"):
            self.absolute_position = False
        if not hasattr(config, "relative_position"):
            self.relative_position = True
        if not hasattr(config, "diag_link"):
            self.diag_link = False
        dims = self.predicate_dims
        self.word_embd = nn.Embedding(self.vocab_size, dims[1])
        self.type_embd = nn.Embedding(self.type_vocab_size, dims[1])
        self.diag = nn.Linear(dims[1], dims[2]) if self.diag_link else None
        if self.absolute_position:
            self.abs_pos_embd = nn.Embedding(512, dims[1])
        if self.relative_position:
            num_relative_positions = 2 * self.max_position_offset + 2
            self.rel_pos_embd = nn.Embedding(num_relative_positions, dims[2])
        if self.mixer_ops[0] is not None and len(self.mixer_ops[0]) > 0:
            self.glob_embd = nn.Embedding(self.glob_size, dims[0])
            self.LayerNorm0 = LayerNorm(dims[0], eps=self.layer_norm_eps)
        self.LayerNorm1 = LayerNorm(dims[1], eps=self.layer_norm_eps)
        self.LayerNorm2 = LayerNorm(dims[2], eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if self.initializer_range < 0:
                init_std = math.sqrt(2.0/sum(list(module.weight.data.shape)))
            else:
                init_std = self.initializer_range
            module.weight.data.normal_(mean=0.0, std=init_std)
        elif isinstance(module, LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.apply(self._init_weights)

    def create_relative_position_ids(self, seq_length):
        r = np.arange(0, seq_length, 1)
        r = np.clip(r, None, self.max_position_offset-1)
        c = np.arange(0, -seq_length, -1)
        c = np.clip(c, -self.max_position_offset+1, None)
        c[1:] += 2 * self.max_position_offset
        rel_pos_ids = toeplitz(c, r)
        return rel_pos_ids

    def mask_relative_position_ids(self, rel_pos_ids, tok_type_ids):
        batch_size = tok_type_ids.shape[0]
        dtype = tok_type_ids.dtype
        device = tok_type_ids.device
        rel_pos_ids = torch.tensor(rel_pos_ids, dtype=dtype, device=device)
        rel_pos_ids = rel_pos_ids.unsqueeze(0).expand(batch_size, -1, -1)
        rel_pos_mask = (tok_type_ids.unsqueeze(-1)==tok_type_ids.unsqueeze(-2))
        rel_pos_mask = rel_pos_mask.to(dtype)
        rel_pos_ids = rel_pos_ids * rel_pos_mask
        rel_pos_ids = rel_pos_ids + (1-rel_pos_mask)*self.max_position_offset
        rel_pos_ids[:, 0, 1:] = 2 * self.max_position_offset
        rel_pos_ids[:, 1:, 0] = 2 * self.max_position_offset + 1
        return rel_pos_ids

    def forward(self, tok_seq=None, tok_type_ids=None):
        input_shape = tok_seq.size()
        device = tok_seq.device
        dtype = tok_seq.dtype
        seq_length = input_shape[1]
        # r=1: unary base predicates & embeddings
        if tok_type_ids is None:
            tok_type_ids = torch.zeros(input_shape, dtype=dtype, device=device)
        tok_embd = self.word_embd(tok_seq)
        type_embd = self.type_embd(tok_type_ids)
        embd1 = tok_embd + type_embd
        if self.absolute_position:
            ape_ids = torch.arange(seq_length, dtype=dtype, device=device)
            ape_ids = ape_ids.unsqueeze(0).expand(input_shape)
            embd1 = embd1 + self.abs_pos_embd(ape_ids)
        # r=2: binary base predicates & embeddings
        if self.relative_position:
            rpe_ids = self.create_relative_position_ids(seq_length)
            rpe_ids = self.mask_relative_position_ids(rpe_ids, tok_type_ids)
            embd2 = self.rel_pos_embd(rpe_ids)
        else:
            shape = (input_shape[:2]) + (input_shape[1], self.predicate_dims[2])
            embd2 = torch.zeros(shape, dtype=embd1.dtype, device=embd1.device)
        if self.diag_link:
            diag2 = self.diag(embd1)
            diag2 = torch.diag_embed(diag2.permute(0, 2, 1), dim1=1, dim2=2)
            embd2 = embd2 + diag2
        # r=0: nullary base predicates & embeddings (global)
        if self.mixer_ops[0] is None or len(self.mixer_ops[0]) == 0:
            embd0 = None
        else:
            glob_ids = torch.arange(self.glob_size, dtype=dtype, device=device)
            glob_ids = glob_ids.unsqueeze(0).expand(input_shape[0], -1)
            embd0 = self.glob_embd(glob_ids)
            embd0 = self.LayerNorm0(embd0)
            embd0 = self.dropout(embd0)
        # apply layernorm and dropout
        embd1 = self.LayerNorm1(embd1)
        embd1 = self.dropout(embd1)
        embd2 = self.LayerNorm2(embd2)
        embd2 = self.dropout(embd2)
        base_predicates = (embd0, embd1, embd2)
        return base_predicates


class LogicOperator(nn.Module):
    def __init__(self, config, arity):
        super().__init__()
        self.arity = arity # output arity
        self.reader = (None, None, None)
        self.writer = None

    def _logic_op(self, input_predicates):
        raise NotImplementedError()
        return output_predicates

    def forward(self, input_predicates):
        predicates = tuple(r(p) for r, p in zip(self.reader, input_predicates))
        predicates = self._logic_op(predicates)
        output_predicates = self.writer(predicates)
        return output_predicates


class GlobalOperator(nn.Module):
    def __init__(self, config, arity):
        super().__init__()
        self.arity = arity # output arity
        self.i_kernel = None
        self.i_premise = None
        self.kernel = None
        self.premise = None
        self.writer = None

    def reader(self, x):
        kernel = self.kernel(x[self.i_kernel])
        premise = self.premise(x[self.i_premise])
        return kernel, premise

    def _logic_op(self, kernel, premise):
        raise NotImplementedError()
        return predicates

    def forward(self, input_predicates):
        kernel, premise = self.reader(input_predicates)
        predicates = self._logic_op(kernel, premise)
        output_predicates = self.writer(predicates)
        return output_predicates


class ConjugateJoinOperator(nn.Module): # based on new operator abstraction
    def __init__(self, config, arity):
        super().__init__()
        self.arity = arity # output arity
        dims = config.predicate_dims
        self.i_kernel = 1
        self.i_premise = 2
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.p_kernel = (0, 2, 3, 1)
        self.p_premise = (0, 3, 1, 2)
        self.p_output = (0, 3, 1, 2)
        self.split_shape = (self.head_size, self.num_heads)
        self.kernel_size = self.head_size * self.num_heads
        self.premise_size = self.head_size
        self.output_size = self.head_size * self.num_heads
        self.kernel = nn.Linear(dims[1], self.kernel_size, bias=False)
        self.premise = nn.Linear(dims[2], self.premise_size)
        self.output = nn.Linear(self.output_size, dims[1])
        self.act_fun = nn.Softmax(dim=-1) # Softmax/Identity
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # dropout/Identity

    def reader(self, x):
        kernel = self.kernel(x[self.i_kernel])
        premise = self.premise(x[self.i_premise])
        if self.i_kernel == 1:
            kernel = kernel.view(kernel.shape[:-1] + self.split_shape)
        if self.i_premise == 1:
            premise = premise.view(premise.shape[:-1] + self.split_shape)
        return kernel, premise

    def writer(self, predicates):
        if self.arity == 1:
            predicates = predicates.contiguous()
            predicates = predicates.view(predicates.shape[:-2] + (-1,))
        predicates = self.output(predicates)
        return predicates

    def forward(self, input_predicates):
        kernel, premise = self.reader(input_predicates)
        kernel = kernel.permute(self.p_kernel)
        kernel = self.act_fun(kernel)
        kernel = self.dropout(kernel)
        premise = premise.permute(self.p_premise)
        predicates = torch.matmul(kernel, premise)
        predicates = predicates.permute(self.p_output)
        output_predicates = self.writer(predicates)
        return output_predicates


class JoinOperator(LogicOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        predicate_dims = config.predicate_dims
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.all_size = self.num_heads * self.head_size
        self.reader = nn.ModuleList([
            nn.Identity(),
            nn.Linear(predicate_dims[1], self.all_size),
            nn.Linear(predicate_dims[2], self.num_heads, bias=False)
        ])
        self.writer = nn.Linear(self.all_size, predicate_dims[1])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _logic_op(self, input_predicates):
        value = input_predicates[1]
        value = value.view(value.shape[:-1] + (self.num_heads, self.head_size))
        value = value.permute(0, 2, 1, 3)
        attention_scores = input_predicates[2]
        attention_scores = attention_scores.permute(0, 3, 1, 2)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        output_predicates = torch.matmul(attention_probs, value)
        output_predicates = output_predicates.permute(0, 2, 1, 3).contiguous()
        new_shape = output_predicates.shape[:-2] + (self.all_size,)
        output_predicates = output_predicates.view(*new_shape)
        return output_predicates


class JoinWithRpeOperator(LogicOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        predicate_dims = config.predicate_dims
        self.max_position_offset = config.max_position_offset
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.all_size = self.num_heads * self.head_size
        self.reader = nn.ModuleList([
            nn.Identity(),
            nn.Linear(predicate_dims[1], self.all_size),
            nn.Linear(predicate_dims[2], self.num_heads, bias=False)
        ])
        self.writer = nn.Linear(self.all_size, predicate_dims[1])
        num_relative_positions = 2 * self.max_position_offset + 2
        self.rel_pos_embd = nn.Embedding(num_relative_positions, self.head_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def create_relative_position_ids(self, seq_length):
        r = np.arange(0, seq_length, 1)
        r = np.clip(r, None, self.max_position_offset-1)
        c = np.arange(0, -seq_length, -1)
        c = np.clip(c, -self.max_position_offset+1, None)
        c[1:] += 2 * self.max_position_offset
        rel_pos_ids = toeplitz(c, r)
        return rel_pos_ids

    def _logic_op(self, input_predicates):
        batch_size = input_predicates[1].shape[0]
        seq_length = input_predicates[1].shape[1]
        value = input_predicates[1]
        value = value.view(value.shape[:-1] + (self.num_heads, self.head_size))
        value = value.permute(0, 2, 1, 3)
        attention_scores = input_predicates[2]
        attention_scores = attention_scores.permute(0, 3, 1, 2)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        predicates = torch.matmul(attention_probs, value)
        rpe_ids = self.create_relative_position_ids(seq_length)
        rpe_ids = torch.tensor(rpe_ids, dtype=torch.long, device=value.device)
        rpe = self.rel_pos_embd(rpe_ids)
        new_shape = (batch_size*self.num_heads, seq_length, seq_length)
        _attention_probs = attention_probs.view(new_shape)
        _attention_probs = _attention_probs.permute(1, 0, 2)
        rpe_bias = torch.matmul(_attention_probs, rpe)
        new_shape = (seq_length, batch_size, self.num_heads, self.head_size)
        rpe_bias = rpe_bias.view(new_shape)
        rpe_bias = rpe_bias.permute(1, 2, 0, 3)
        predicates = predicates + rpe_bias
        output_predicates = predicates.permute(0, 2, 1, 3).contiguous()
        new_shape = output_predicates.shape[:-2] + (self.all_size,)
        output_predicates = output_predicates.view(*new_shape)
        return output_predicates


class ScatterOperator(GlobalOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 1
        dims = config.predicate_dims
        self.i_kernel = 1
        self.i_premise = 0
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.glob_size = config.glob_size
        self.all_size = self.num_heads * self.head_size
        self.kernel_size = self.num_heads * self.glob_size
        self.premise_size = self.num_heads * self.head_size
        self.kernel = nn.Linear(dims[1], self.kernel_size, bias=False)
        self.premise = nn.Linear(dims[0], self.premise_size, bias=False)
        self.writer = nn.Linear(self.all_size, dims[1])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _logic_op(self, kernel, premise):
        new_shape = kernel.shape[:-1] + (self.num_heads, self.glob_size)
        kernel = kernel.view(new_shape)
        kernel = kernel.permute(0, 2, 1, 3)
        kernel = nn.Softmax(dim=-1)(kernel)
        kernel = self.dropout(kernel)
        new_shape = premise.shape[:-1] + (self.num_heads, self.head_size)
        premise = premise.view(new_shape)
        premise = premise.permute(0, 2, 1, 3)
        predicates = torch.matmul(kernel, premise)
        predicates = predicates.permute(0, 2, 1, 3).contiguous()
        predicates = predicates.view(predicates.shape[:-2]+(-1,))
        return predicates


class Scatter2Operator(GlobalOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 2
        dims = config.predicate_dims
        self.i_kernel = 2
        self.i_premise = 0
        self.glob_size = config.glob_size
        self.num_heads = int(max(dims[2]/self.glob_size, 1.0))
        self.head_size = int(dims[2]/self.num_heads)
        self.kernel_size = self.num_heads * self.glob_size
        self.premise_size = self.num_heads * self.head_size
        self.all_size = self.num_heads * self.head_size
        self.kernel = nn.Linear(dims[2], self.kernel_size, bias=False)
        self.premise = nn.Linear(dims[0], self.premise_size, bias=False)
        self.writer = nn.Linear(self.all_size, dims[2])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _logic_op(self, kernel, premise):
        orig_shape = kernel.shape
        new_shape = (kernel.shape[0], -1, self.num_heads, self.glob_size)
        kernel = kernel.view(new_shape)
        kernel = kernel.permute(0, 2, 1, 3)
        kernel = nn.Softmax(dim=-1)(kernel)
        kernel = self.dropout(kernel)
        new_shape = premise.shape[:-1] + (self.num_heads, self.head_size)
        premise = premise.view(new_shape)
        premise = premise.permute(0, 2, 1, 3)
        predicates = torch.matmul(kernel, premise)
        predicates = predicates.permute(0, 2, 1, 3)
        predicates = predicates.contiguous()
        predicates = predicates.view(orig_shape[:-1] + (-1,))
        return predicates


class GatherOperator(GlobalOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 0
        dims = config.predicate_dims
        self.i_kernel = 1
        self.i_premise = 1
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.glob_size = config.glob_size
        self.kernel_size = self.num_heads * self.glob_size
        self.premise_size = self.num_heads * self.head_size
        self.all_size = self.num_heads * self.head_size
        self.kernel = nn.Linear(dims[1], self.kernel_size, bias=False)
        self.premise = nn.Linear(dims[1], self.premise_size, bias=False)
        self.writer = nn.Linear(self.all_size, dims[0])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _logic_op(self, kernel, premise):
        new_shape = kernel.shape[:-1] + (self.num_heads, self.glob_size)
        kernel = kernel.view(new_shape)
        kernel = kernel.permute(0, 2, 3, 1)
        kernel = nn.Softmax(dim=-1)(kernel)
        kernel = self.dropout(kernel)
        new_shape = premise.shape[:-1] + (self.num_heads, self.head_size)
        premise = premise.view(new_shape)
        premise = premise.permute(0, 2, 1, 3)
        predicates = torch.matmul(kernel, premise)
        predicates = predicates.permute(0, 2, 1, 3).contiguous()
        predicates = predicates.view(predicates.shape[:-2]+(-1,))
        return predicates


class Gather2Operator(GlobalOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 0
        dims = config.predicate_dims
        self.i_kernel = 2
        self.i_premise = 2
        self.glob_size = config.glob_size
        self.num_heads = int(max(dims[2]/self.glob_size, 1.0))
        self.head_size = int(dims[2]/self.num_heads)
        self.kernel_size = self.num_heads * self.glob_size
        self.premise_size = self.num_heads * self.head_size
        self.all_size = self.num_heads * self.head_size
        self.kernel = nn.Linear(dims[2], self.kernel_size, bias=False)
        self.premise = nn.Linear(dims[2], self.premise_size, bias=False)
        self.writer = nn.Linear(self.all_size, dims[0])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _logic_op(self, kernel, premise):
        new_shape = (kernel.shape[0], -1, self.num_heads, self.glob_size)
        kernel = kernel.view(new_shape)
        kernel = kernel.permute(0, 2, 3, 1)
        kernel = nn.Softmax(dim=-1)(kernel)
        kernel = self.dropout(kernel)
        new_shape = (premise.shape[0], -1, self.num_heads, self.head_size)
        premise = premise.view(new_shape)
        premise = premise.permute(0, 2, 1, 3)
        predicates = torch.matmul(kernel, premise)
        predicates = predicates.permute(0, 2, 1, 3)
        predicates = predicates.contiguous()
        predicates = predicates.view(predicates.shape[:-2] + (-1,))
        return predicates


class LambdaOperator(LogicOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 2
        predicate_dims = config.predicate_dims
        self.reader = nn.ModuleList([
            nn.Linear(predicate_dims[arity-1], predicate_dims[arity])
            if r == arity-1 else nn.Identity()
            for r in range(len(config.mixer_ops))
        ])
        self.writer = nn.Identity()

    def _logic_op(self, input_predicates):
        predicates = input_predicates[self.arity-1]
        new_shape = input_predicates[self.arity].shape
        output_predicates = predicates.unsqueeze(-2).expand(*new_shape)
        return output_predicates


class ExpandOperator(LogicOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 2
        predicate_dims = config.predicate_dims
        self.max_span = config.max_span
        self.span_dim = predicate_dims[2]
        self.all_size = self.max_span * self.span_dim
        self.reader = nn.ModuleList([
            nn.Identity(),
            nn.Linear(predicate_dims[1], self.all_size),
            nn.Identity()
        ])
        self.writer = nn.Identity()

    def _logic_op(self, input_predicates):
        predicates = input_predicates[1]
        batch_size = predicates.shape[0]
        seq_length = predicates.shape[1]
        device = predicates.device
        dtype = predicates.dtype
        new_shape = (batch_size, seq_length, self.max_span, self.span_dim)
        predicates = predicates.view(new_shape)
        _t = torch.arange(seq_length, dtype=torch.long, device=device)
        _s = torch.arange(self.max_span, dtype=torch.long, device=device)
        scatter_index = (_t.unsqueeze(-1) + _s.unsqueeze(0)) % seq_length
        scatter_index = scatter_index.unsqueeze(-1).unsqueeze(0)
        scatter_index = scatter_index.expand(batch_size, -1, -1, self.span_dim)
        target_shape = (batch_size, seq_length, seq_length, self.span_dim)
        _zeros = torch.zeros(target_shape, device=device, dtype=dtype)
        output_predicates = _zeros.scatter_(2, scatter_index, predicates)
        return output_predicates


class ReduceOperator(LogicOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 1
        predicate_dims = config.predicate_dims
        self.max_span = config.max_span
        self.reader = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity()
        ])
        self.writer = nn.Linear(
            self.max_span*predicate_dims[2],
            predicate_dims[1]
        )

    def _logic_op(self, input_predicates):
        predicates = input_predicates[2]
        batch_size = predicates.shape[0]
        seq_length = predicates.shape[1]
        predicate_dim = predicates.shape[-1]
        device = predicates.device
        dtype = predicates.dtype
        _t = torch.arange(seq_length, dtype=torch.long, device=device)
        _s = torch.arange(self.max_span, dtype=torch.long, device=device)
        gather_index = (_t.unsqueeze(-1) + _s.unsqueeze(0)) % seq_length
        gather_index = gather_index.unsqueeze(-1).unsqueeze(0)
        gather_index = gather_index.expand(batch_size, -1, -1, predicate_dim)
        predicates = predicates.gather(2, gather_index)
        output_predicates = predicates.view(predicates.shape[:2] + (-1,))
        return output_predicates


class MuOperator(LogicOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 1
        predicate_dims = config.predicate_dims
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.all_size = self.num_heads * self.head_size
        self.reader = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Linear(
                predicate_dims[2],
                self.num_heads+self.head_size,
                bias=False
            )
        ])
        self.writer = nn.Linear(self.all_size, predicate_dims[1])

    def _logic_op(self, input_predicates):
        predicates = input_predicates[2]
        query = predicates[:, :, :, :self.num_heads]
        key = predicates[:, :, :, self.num_heads:]
        query = query.transpose(-1, -2)
        query = nn.Softmax(dim=-1)(query)
        predicates = torch.matmul(query, key)
        output_shape = predicates.shape[:-2] + (-1,)
        output_predicates = predicates.view(output_shape)
        return output_predicates


class QuantificationOperator(LogicOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity < 2
        predicate_dims = config.predicate_dims
        self.reader = nn.ModuleList([
            nn.Linear(predicate_dims[r], predicate_dims[r])
            if r == arity + 1 else nn.Identity()
            for r in range(len(config.mixer_ops))
        ])
        self.writer = nn.Linear(predicate_dims[arity+1], predicate_dims[arity])

    def _logic_op(self, input_predicates):
        predicates = input_predicates[self.arity+1]
        output_predicates = predicates.max(dim=-2)[0]
        return output_predicates


class AssociativeOperator(LogicOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 2
        predicate_dims = config.predicate_dims
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.all_size = self.num_heads * self.head_size
        self.reader = nn.ModuleList([
            nn.Identity(),
            nn.Linear(predicate_dims[1], self.all_size*2),
            nn.Identity()
        ])
        self.writer = nn.Linear(self.num_heads, predicate_dims[2])

    def transpose_for_association(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.transpose(-2, -3)

    def _logic_op(self, input_predicates):
        unary_predicates = input_predicates[1]
        scaling = math.sqrt(math.sqrt(self.head_size))
        query = unary_predicates[:, :, :self.all_size] / scaling
        query = self.transpose_for_association(query)
        key = unary_predicates[:, :, self.all_size:] / scaling
        key = self.transpose_for_association(key)
        predicates = torch.matmul(query, key.transpose(-1, -2))
        output_predicates = predicates.permute(0, 2, 3, 1)
        return output_predicates


class AssociativeWithSoftmaxOperator(LogicOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 2
        predicate_dims = config.predicate_dims
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.all_size = self.num_heads * self.head_size
        self.reader = nn.ModuleList([
            nn.Identity(),
            nn.Linear(predicate_dims[1], self.all_size*2, bias=False),
            nn.Identity()
        ])
        self.writer = nn.Linear(self.num_heads, predicate_dims[2])

    def transpose_for_association(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.transpose(-2, -3)

    def _logic_op(self, input_predicates):
        unary_predicates = input_predicates[1]
        query = unary_predicates[:, :, :self.all_size]
        query = self.transpose_for_association(query)
        query = nn.Softmax(dim=-1)(query)
        key = unary_predicates[:, :, self.all_size:]
        key = self.transpose_for_association(key)
        predicates = torch.matmul(query, key.transpose(-1, -2))
        output_predicates = predicates.permute(0, 2, 3, 1)
        return output_predicates


class AssociativeWithRpeOperator(LogicOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 2
        predicate_dims = config.predicate_dims
        self.max_position_offset = config.max_position_offset
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.all_size = self.num_heads * self.head_size
        self.reader = nn.ModuleList([
            nn.Identity(),
            nn.Linear(predicate_dims[1], self.all_size*2),
            nn.Identity()
        ])
        self.writer = nn.Linear(self.num_heads, predicate_dims[2])
        num_relative_positions = 2 * self.max_position_offset + 2
        self.rel_pos_embd = nn.Embedding(num_relative_positions, self.head_size)

    def transpose_for_association(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.transpose(-2, -3)

    def create_relative_position_ids(self, seq_length):
        r = np.arange(0, seq_length, 1)
        r = np.clip(r, None, self.max_position_offset-1)
        c = np.arange(0, -seq_length, -1)
        c = np.clip(c, -self.max_position_offset+1, None)
        c[1:] += 2 * self.max_position_offset
        rel_pos_ids = toeplitz(c, r)
        return rel_pos_ids

    def _logic_op(self, input_predicates):
        batch_size = input_predicates[1].shape[0]
        seq_length = input_predicates[1].shape[1]
        unary_predicates = input_predicates[1]
        scaling = math.sqrt(math.sqrt(self.head_size))
        query = unary_predicates[:, :, :self.all_size] / scaling
        query = self.transpose_for_association(query)
        key = unary_predicates[:, :, self.all_size:] / scaling
        key = self.transpose_for_association(key)
        predicates = torch.matmul(query, key.transpose(-1, -2))
        rpe_ids = self.create_relative_position_ids(seq_length)
        rpe_ids = torch.tensor(rpe_ids, dtype=torch.long, device=query.device)
        rpe = self.rel_pos_embd(rpe_ids) / scaling
        new_shape = (batch_size*self.num_heads, seq_length, self.head_size)
        _query = query.contiguous()
        _query = _query.view(new_shape)
        _query = _query.permute(1, 0, 2)
        rpe_bias = torch.matmul(rpe, _query.transpose(-1, -2))
        new_shape = (seq_length, seq_length, batch_size, self.num_heads)
        rpe_bias = rpe_bias.view(new_shape)
        rpe_bias = rpe_bias.permute(2, 3, 0, 1)
        predicates = predicates + rpe_bias
        output_predicates = predicates.permute(0, 2, 3, 1)
        return output_predicates


class AssociativeWithLayerNormOperator(GlobalOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 2
        dims = config.predicate_dims
        self.i_kernel = 1
        self.i_premise = 1
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.kernel_size = self.num_heads * self.head_size
        self.premise_size = self.num_heads * self.head_size
        self.all_size = self.num_heads
        self.kernel = nn.Linear(dims[1], self.kernel_size)
        self.premise = nn.Linear(dims[1], self.premise_size)
        self.writer = nn.Linear(self.all_size, dims[2])
        self.LayerNorm = LayerNorm(self.head_size, eps=config.layer_norm_eps)

    def transpose_for_association(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.transpose(-2, -3)

    def _logic_op(self, kernel, premise):
        kernel = self.transpose_for_association(kernel)
        kernel = self.LayerNorm(kernel)
        premise = self.transpose_for_association(premise)
        predicates = torch.matmul(kernel, premise.transpose(-1, -2))
        predicates = predicates.permute(0, 2, 3, 1)
        return predicates


class LinkOperator(GlobalOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 1
        dims = config.predicate_dims
        self.i_kernel = 0
        self.i_premise = 1
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.glob_size = config.glob_size
        self.kernel_size = self.num_heads * self.head_size
        self.premise_size = self.num_heads * self.head_size
        self.all_size = self.num_heads * self.glob_size
        self.kernel = nn.Linear(dims[0], self.kernel_size)
        self.premise = nn.Linear(dims[1], self.premise_size)
        self.writer = nn.Linear(self.all_size, dims[1])

    def transpose_for_association(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.transpose(-2, -3)

    def _logic_op(self, kernel, premise):
        scaling = math.sqrt(math.sqrt(self.head_size))
        kernel = kernel / scaling
        kernel = self.transpose_for_association(kernel)
        premise = premise / scaling
        premise = self.transpose_for_association(premise)
        predicates = torch.matmul(kernel, premise.transpose(-1, -2))
        predicates = predicates.permute(0, 3, 1, 2).contiguous()
        predicates = predicates.view(predicates.shape[:-2] + (-1,))
        return predicates

class LinkWithLayerNormOperator(GlobalOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 1
        dims = config.predicate_dims
        self.i_kernel = 0
        self.i_premise = 1
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.glob_size = config.glob_size
        self.kernel_size = self.num_heads * self.head_size
        self.premise_size = self.num_heads * self.head_size
        self.all_size = self.num_heads * self.glob_size
        self.kernel = nn.Linear(dims[0], self.kernel_size)
        self.premise = nn.Linear(dims[1], self.premise_size)
        self.writer = nn.Linear(self.all_size, dims[1])
        self.LayerNorm = LayerNorm(self.head_size, eps=config.layer_norm_eps)

    def transpose_for_association(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.transpose(-2, -3)

    def _logic_op(self, kernel, premise):
        kernel = self.transpose_for_association(kernel)
        kernel = self.LayerNorm(kernel)
        premise = self.transpose_for_association(premise)
        predicates = torch.matmul(kernel, premise.transpose(-1, -2))
        predicates = predicates.permute(0, 3, 1, 2).contiguous()
        predicates = predicates.view(predicates.shape[:-2] + (-1,))
        return predicates



class Link2Operator(GlobalOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 2
        dims = config.predicate_dims
        self.i_kernel = 0
        self.i_premise = 2
        self.glob_size = config.glob_size
        self.num_heads = int(max(dims[2]/self.glob_size, 1.0))
        self.head_size = dims[2]
        self.kernel_size = self.num_heads * self.head_size
        self.premise_size = self.head_size
        self.all_size = self.num_heads * self.glob_size
        self.kernel = nn.Linear(dims[0], self.kernel_size)
        self.premise = nn.Linear(dims[2], self.premise_size)
        self.writer = nn.Linear(self.all_size, dims[2])

    def transpose_for_association(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.transpose(-2, -3)

    def _logic_op(self, kernel, premise):
        orig_shape = premise.shape
        scaling = math.sqrt(math.sqrt(self.head_size))
        kernel = kernel / scaling
        kernel = self.transpose_for_association(kernel)
        kernel = kernel.contiguous()
        kernel = kernel.view(kernel.shape[0], -1, kernel.shape[-1])
        premise = premise / scaling
        premise = premise.view(premise.shape[0], -1, premise.shape[-1])
        predicates = torch.matmul(kernel, premise.transpose(-1, -2))
        predicates = predicates.permute(0, 2, 1)
        predicates = predicates.view(orig_shape[:-1] + (-1,))
        return predicates

class Link2WithLayerNormOperator(GlobalOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 2
        dims = config.predicate_dims
        self.i_kernel = 0
        self.i_premise = 2
        self.glob_size = config.glob_size
        self.num_heads = int(max(dims[2]/self.glob_size, 1.0))
        self.head_size = dims[2]
        self.kernel_size = self.num_heads * self.head_size
        self.premise_size = self.head_size
        self.all_size = self.num_heads * self.glob_size
        self.kernel = nn.Linear(dims[0], self.kernel_size)
        self.premise = nn.Linear(dims[2], self.premise_size)
        self.writer = nn.Linear(self.all_size, dims[2])
        self.LayerNorm = LayerNorm(self.head_size, eps=config.layer_norm_eps)

    def transpose_for_association(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.transpose(-2, -3)

    def _logic_op(self, kernel, premise):
        orig_shape = premise.shape
        kernel = self.transpose_for_association(kernel)
        kernel = kernel.contiguous()
        kernel = kernel.view(kernel.shape[0], -1, kernel.shape[-1])
        kernel = self.LayerNorm(kernel)
        premise = premise.view(premise.shape[0], -1, premise.shape[-1])
        predicates = torch.matmul(kernel, premise.transpose(-1, -2))
        predicates = predicates.permute(0, 2, 1)
        predicates = predicates.view(orig_shape[:-1] + (-1,))
        return predicates



class ReflectionOperator(GlobalOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 0
        dims = config.predicate_dims
        self.i_kernel = 0
        self.i_premise = 0
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.glob_size = config.glob_size
        self.kernel_size = self.num_heads * self.head_size
        self.premise_size = self.num_heads * self.head_size
        self.all_size = 2 * self.num_heads * self.glob_size
        self.kernel = nn.Linear(dims[0], self.kernel_size)
        self.premise = nn.Linear(dims[0], self.premise_size)
        self.writer = nn.Linear(self.all_size, dims[0])

    def transpose_for_association(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.transpose(-2, -3)

    def _logic_op(self, kernel, premise):
        scaling = math.sqrt(math.sqrt(self.head_size))
        kernel = kernel / scaling
        kernel = self.transpose_for_association(kernel)
        premise = premise / scaling
        premise = self.transpose_for_association(premise)
        predicates = torch.matmul(kernel, premise.transpose(-1, -2))
        p0 = predicates.permute(0, 3, 1, 2).contiguous()
        p1 = predicates.permute(0, 2, 1, 3).contiguous()
        predicates = torch.cat((p0, p1), dim=-1)
        predicates = predicates.view(predicates.shape[:-2] + (-1,))
        return predicates

class ReflectionWithLayerNormOperator(GlobalOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 0
        dims = config.predicate_dims
        self.i_kernel = 0
        self.i_premise = 0
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.glob_size = config.glob_size
        self.kernel_size = self.num_heads * self.head_size
        self.premise_size = self.num_heads * self.head_size
        self.all_size = 2 * self.num_heads * self.glob_size
        self.kernel = nn.Linear(dims[0], self.kernel_size)
        self.premise = nn.Linear(dims[0], self.premise_size)
        self.writer = nn.Linear(self.all_size, dims[0])
        self.LayerNorm = LayerNorm(self.head_size, eps=config.layer_norm_eps)

    def transpose_for_association(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.transpose(-2, -3)

    def _logic_op(self, kernel, premise):
        kernel = self.transpose_for_association(kernel)
        kernel = self.LayerNorm(kernel)
        premise = self.transpose_for_association(premise)
        predicates = torch.matmul(kernel, premise.transpose(-1, -2))
        p0 = predicates.permute(0, 3, 1, 2).contiguous()
        p1 = predicates.permute(0, 2, 1, 3).contiguous()
        predicates = torch.cat((p0, p1), dim=-1)
        predicates = predicates.view(predicates.shape[:-2] + (-1,))
        return predicates



class FuseOperator(GlobalOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 0
        dims = config.predicate_dims
        self.i_kernel = 0
        self.i_premise = 0
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.glob_size = config.glob_size
        self.kernel_size = self.num_heads * self.glob_size
        self.premise_size = self.num_heads * self.head_size
        self.all_size = self.num_heads * self.head_size
        self.kernel = nn.Linear(dims[0], self.kernel_size, bias=False)
        self.premise = nn.Linear(dims[0], self.premise_size, bias=False)
        self.writer = nn.Linear(self.all_size, dims[0])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _logic_op(self, kernel, premise):
        new_shape = kernel.shape[:-1] + (self.num_heads, self.glob_size)
        kernel = kernel.view(new_shape)
        kernel = kernel.permute(0, 2, 1, 3)
        kernel = nn.Softmax(dim=-1)(kernel)
        kernel = self.dropout(kernel)
        new_shape = premise.shape[:-1] + (self.num_heads, self.head_size)
        premise = premise.view(new_shape)
        premise = premise.permute(0, 2, 1, 3)
        predicates = torch.matmul(kernel, premise)
        predicates = predicates.permute(0, 2, 1, 3).contiguous()
        predicates = predicates.view(predicates.shape[:-2]+(-1,))
        return predicates


class ProductOperator(LogicOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 2
        predicate_dims = config.predicate_dims
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.all_size = self.num_heads * self.head_size
        self.reader = nn.ModuleList([
            nn.Identity(),
            nn.Linear(predicate_dims[1], self.all_size),
            nn.Linear(predicate_dims[2], self.head_size)
        ])
        self.writer = nn.Linear(self.num_heads, predicate_dims[2])

    def _logic_op(self, input_predicates):
        unary_predicates = input_predicates[1]
        binary_predicates = input_predicates[2]
        scaling = math.sqrt(math.sqrt(self.head_size))
        query = unary_predicates / scaling
        query = query.view(query.shape[:-1]+(self.num_heads, self.head_size))
        key = binary_predicates / scaling
        predicates = torch.matmul(query, key.transpose(-1, -2))
        output_predicates = predicates.permute(0, 1, 3, 2)
        return output_predicates


class ProductWithSoftmaxOperator(LogicOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 2
        predicate_dims = config.predicate_dims
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.all_size = self.num_heads * self.head_size
        self.reader = nn.ModuleList([
            nn.Identity(),
            nn.Linear(predicate_dims[1], self.all_size, bias=False),
            nn.Linear(predicate_dims[2], self.head_size, bias=False)
        ])
        self.writer = nn.Linear(self.num_heads, predicate_dims[2])

    def _logic_op(self, input_predicates):
        unary_predicates = input_predicates[1]
        binary_predicates = input_predicates[2]
        query = unary_predicates
        query = query.view(query.shape[:-1]+(self.num_heads, self.head_size))
        query_ = nn.Softmax(dim=-1)(query)
        key = binary_predicates
        key_ = nn.Softmax(dim=-1)(key)
        predicates = torch.matmul(query_, key.transpose(-1, -2))
        predicates = predicates + torch.matmul(query, key_.transpose(-1, -2))
        output_predicates = predicates.permute(0, 1, 3, 2)
        return output_predicates


class ProductWithLayerNormOperator(GlobalOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 2
        dims = config.predicate_dims
        self.i_kernel = 1
        self.i_premise = 2
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.kernel_size = self.num_heads * self.head_size
        self.premise_size = self.head_size
        self.all_size = self.num_heads
        self.kernel = nn.Linear(dims[1], self.kernel_size)
        self.premise = nn.Linear(dims[2], self.premise_size)
        self.writer = nn.Linear(self.all_size, dims[2])
        self.LayerNorm = LayerNorm(self.head_size, eps=config.layer_norm_eps)

    def _logic_op(self, kernel, premise):
        kernel = kernel.view(kernel.shape[:-1]+(self.num_heads, self.head_size))
        kernel = self.LayerNorm(kernel)
        predicates = torch.matmul(kernel, premise.transpose(-1, -2))
        predicates = predicates.permute(0, 1, 3, 2)
        return predicates


class CompositionOperator(LogicOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 2
        predicate_dims = config.predicate_dims
        self.all_size = predicate_dims[2]
        self.reader = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Linear(predicate_dims[2], predicate_dims[2]*2, bias=False)
        ])
        self.writer = nn.Linear(predicate_dims[2], predicate_dims[2])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _logic_op(self, input_predicates):
        binary_predicates = input_predicates[2]
        query = binary_predicates[:, :, :, :self.all_size]
        query = query.permute(0, 3, 1, 2)
        key = binary_predicates[:, :, :, self.all_size:]
        key = key.permute(0, 3, 1, 2)
        query = nn.Softmax(dim=-1)(query)
        query = self.dropout(query)
        binary_predicates = torch.matmul(query, key)
        output_predicates = binary_predicates.permute(0, 2, 3, 1)
        return output_predicates


class ChartOperator(LogicOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 1
        predicate_dims = config.predicate_dims
        self.max_span = config.max_span
        self.span_dim = config.span_dim
        self.all_size = self.max_span * self.span_dim
        self.reader = nn.ModuleList([
            nn.Identity(),
            nn.Linear(predicate_dims[1], self.all_size*2),
            nn.Identity()
        ])
        self.writer = nn.Linear(self.all_size, predicate_dims[1])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        fwd, bwd = self._gather_index(self.max_span)
        fwd = fwd.view((1,1)+fwd.shape+(1,))
        bwd = bwd.view((1,1)+bwd.shape+(1,))
        self.register_buffer("fwd", fwd)
        self.register_buffer("bwd", bwd)

    def _gather_index(self, D):
        fwd = [
            [tau-t if tau-t >= 0 and tau-t < D else D for tau in range(2*D)]
            for t in range(D)
        ]
        bwd = [[t+delta for delta in range(D)] for t in range(D)]
        return torch.tensor(fwd), torch.tensor(bwd)

    def _gather(self, x, index, padding=0.0):
        pad_shape = x.shape[:2] + (1, x.shape[-1])
        pad = torch.full((1,1,1,1), padding, device=x.device, dtype=x.dtype)
        pad = pad.expand(*pad_shape)
        x = torch.cat((x, pad), dim=2)
        x = x.view((x.shape[0], -1, self.max_span) + x.shape[2:])
        x = x.gather(3, index.expand(x.shape[:2] + (-1,-1) + x.shape[4:]))
        return x

    def _logic_op(self, input_predicates):
        input_shape = input_predicates[1].shape
        split_shape = input_shape[:-1] + (2, self.max_span, self.span_dim)
        output_shape = input_shape[:-1] + (self.all_size,)
        P = input_predicates[1].view(*split_shape)
        L = P[:, :, 0, :, :]
        R = P[:, :, 1, :, :]
        R = R.roll(shifts=-1, dims=1)
        L = self._gather(L, self.fwd, -1e4)
        R = self._gather(R, self.fwd.roll(shifts=1, dims=3), 0.0)
        L = L.permute(0,1,4,2,3)
        R = R.permute(0,1,4,2,3)
        L = nn.Softmax(dim=-1)(L)
        L0 = L[:, :, :, :, :self.max_span]
        L1 = L[:, :, :, :, self.max_span:]
        R0 = R[:, :, :, :, :self.max_span]
        R1 = R[:, :, :, :, self.max_span:]
        Q0 = torch.matmul(L0, R0)
        Q1 = torch.matmul(L0, R1)
        Q1 = Q1 + torch.matmul(L1, R0.roll(shifts=-1, dims=1))
        mask = [1]*(Q1.shape[1]-1) + [0]
        mask = torch.tensor(mask, device=Q1.device, dtype=Q1.dtype)
        Q1 = Q1 * mask.view((1, mask.shape[0], 1, 1, 1))
        Q = torch.cat((Q0, Q1), dim=-1)
        Q = Q.permute(0, 1, 3, 4, 2)
        Q = Q.gather(3, self.bwd.expand(Q.shape[:2] + (-1,-1) + Q.shape[4:]))
        output_predicates = Q.view(*output_shape)
        return output_predicates


class HeadOperator(LogicOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 1
        predicate_dims = config.predicate_dims
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.all_size = self.num_heads * self.head_size
        self.reader = nn.ModuleList([
            nn.Identity(),
            nn.Linear(
                predicate_dims[1],
                self.num_heads**2+self.all_size,
                bias=False
            ),
            nn.Identity()
        ])
        self.writer = nn.Linear(self.all_size, predicate_dims[1])

    def _logic_op(self, input_predicates):
        predicates = input_predicates[1]
        query = predicates[:, :, :self.num_heads**2]
        query = query.view(query.shape[:-1] + (self.num_heads, self.num_heads))
        query = F.relu(query)
        key = predicates[:, :, self.num_heads**2:]
        key = key.view(key.shape[:-1] + (self.num_heads, self.head_size))
        predicates = torch.matmul(query, key)
        output_predicates = predicates.view(predicates.shape[:-2]+(-1,))
        return output_predicates


class Head2Operator(LogicOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 2
        predicate_dims = config.predicate_dims
        self.num_heads = int(math.sqrt(predicate_dims[2]))
        self.head_size = int(math.sqrt(predicate_dims[2]))
        self.all_size = self.num_heads * self.head_size
        self.reader = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Linear(predicate_dims[2], 2*self.all_size, bias=False),
        ])
        self.writer = nn.Linear(self.all_size, predicate_dims[2])

    def _logic_op(self, input_predicates):
        predicates = input_predicates[2]
        query = predicates[:, :, :, :self.all_size]
        query = query.view(query.shape[:-1] + (self.num_heads, self.num_heads))
        query = F.relu(query)
        key = predicates[:, :, :, self.all_size:]
        key = key.view(key.shape[:-1] + (self.num_heads, self.head_size))
        predicates = torch.matmul(query, key)
        output_predicates = predicates.view(predicates.shape[:-2]+(-1,))
        return output_predicates


class ReGluOperator(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.gating = nn.Linear(in_features, out_features)
        self.direct = nn.Linear(in_features, out_features)

    def forward(self, x):
        g = self.gating(x)
        x = self.direct(x)
        y = F.relu(g) * x
        return y


BOOLEAN = {
    "mlp": nn.Linear,
    "reglu": ReGluOperator
}


class BooleanOperator(LogicOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        predicate_dims = config.predicate_dims
        intermediate_dims = config.intermediate_dims
        if not hasattr(config, "boolean_type"):
            config.boolean_type = "mlp"
        self.reader = nn.ModuleList([
            nn.Identity()
            for r in range(len(config.mixer_ops))
        ])
        self.clause_in = BOOLEAN[config.boolean_type](
            max(arity, 1)*predicate_dims[arity],
            intermediate_dims[arity]
        )
        self.clause_out = nn.Linear(
            intermediate_dims[arity],
            predicate_dims[arity]
        )
        self.clause_skip = nn.Linear(
            predicate_dims[2],
            predicate_dims[2]
        ) if self.arity == 2 else None
        self.writer = nn.Identity()
        self.act_fun = (
            ACT2FN[config.hidden_act]
            if config.boolean_type == "mlp"
            else nn.Identity()
        )

    def _logic_op(self, input_predicates):
        predicates = input_predicates[self.arity]
        if self.arity == 2:
            predicates_T = predicates.transpose(1, 2)
            predicates = torch.cat((predicates, predicates_T), dim=3)
            skip = self.clause_skip(predicates_T)
        predicates = self.clause_in(predicates)
        predicates = self.act_fun(predicates) # drop if ReGLU is used
        predicates = self.clause_out(predicates)
        if self.arity == 2:
            predicates = predicates + skip
        output_predicates = predicates
        return output_predicates


class PermutationOperator(LogicOperator):
    def __init__(self, config, arity):
        super().__init__(config, arity)
        assert arity == 2 # only support binary for now
        predicate_dims = config.predicate_dims
        self.reader = nn.ModuleList([
            nn.Identity() for _ in range(len(config.mixer_ops))
        ])
        self.writer = nn.Linear(predicate_dims[arity], predicate_dims[arity])

    def _logic_op(self, input_predicates):
        output_predicates = input_predicates[self.arity].transpose(1, 2)
        return output_predicates


OPS = {
    "cjoin": ConjugateJoinOperator,
    "join": JoinOperator,
    "join_": JoinWithRpeOperator,
    "scatter": ScatterOperator,
    "scatter2": Scatter2Operator,
    "gather": GatherOperator,
    "gather2": Gather2Operator,
    "lambda": LambdaOperator,
    "expand": ExpandOperator,
    "mu": MuOperator,
    "quantif": QuantificationOperator,
    "reduce": ReduceOperator,
    "assoc": AssociativeOperator,
    "assoc_": AssociativeWithRpeOperator,
    "assoc+": AssociativeWithSoftmaxOperator,
    "assoc*": AssociativeWithLayerNormOperator,
    "link": LinkOperator,
    "link*": LinkWithLayerNormOperator,
    "link2": Link2Operator,
    "link2*": Link2WithLayerNormOperator,
    "reflec": ReflectionOperator,
    "reflec*": ReflectionWithLayerNormOperator,
    "fuse": FuseOperator,
    "prod": ProductOperator,
    "prod+": ProductWithSoftmaxOperator,
    "prod*": ProductWithLayerNormOperator,
    "trans": CompositionOperator,
    "head": HeadOperator,
    "head2": Head2Operator,
    "chart": ChartOperator,
    "perm": PermutationOperator,
    "bool": BooleanOperator,
}


class Deduction(nn.Module):
    def __init__(self, config, deduction_type=None):
        super().__init__()
        predicate_dims = config.predicate_dims
        if config.mixer_ops[0] is None or len(config.mixer_ops[0])==0:
            ops_keys = {
                "object": config.mixer_ops,
                "permut": {0:None, 1:None, 2:("perm",)},
                "predic": {0:None, 1:("bool",), 2:("bool",)}
            }[deduction_type]
        else:
            ops_keys = {
                "object": config.mixer_ops,
                "permut": {0:None, 1:None, 2:("perm",)},
                "predic": {0:("bool",), 1:("bool",), 2:("bool",)}
            }[deduction_type]
        self.normalizers = nn.ModuleList(
            LayerNorm(
                predicate_dims[r],
                eps=config.layer_norm_eps,
                elementwise_affine=False
            )
            if (
                config.mixer_ops[r] is not None
                and not (deduction_type == "permut" and r == 1)
            ) else nn.Identity()
            for r in range(len(config.mixer_ops))
        )
        self.operators = nn.ModuleList(
            nn.ModuleList(OPS[s](config, r) for s in ops)
            if ops is not None else None
            for r, ops in ops_keys.items()
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_predicates):
        predicates = tuple(
            f(p) for f, p in zip(self.normalizers, input_predicates)
        )
        predicates = tuple(
            sum(op(predicates) for op in ops)
            if ops is not None else None
            for ops in self.operators
        )
        output_predicates = tuple(
            self.dropout(p)
            if p is not None else None
            for p in predicates
        )
        return output_predicates


class ReasonerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.deducers = nn.ModuleList(
            Deduction(config, s)
            # for s in ("object", "permut", "predic")
            for s in ("object", "predic")
        )

    def forward(self, input_predicates):
        predicates = input_predicates
        for deducer in self.deducers:
            skips = predicates
            predicates = deducer(predicates)
            predicates = tuple(
                p + s if p is not None else s
                for p, s in zip(predicates, skips)
            )
        output_predicates = predicates
        return output_predicates


class Reasoner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            ReasonerLayer(config)
            for _ in range(config.num_layers)
        )
        self.normalizers = nn.ModuleList(
            LayerNorm(config.predicate_dims[r], eps=config.layer_norm_eps)
            if config.mixer_ops[r] is not None else nn.Identity()
            for r in range(len(config.mixer_ops))
        )

    def forward(self, input_predicates):
        predicates = input_predicates
        for layer in self.layers:
            predicates = layer(predicates)
        predicates = (f(p) for f, p in zip(self.normalizers, predicates))
        output_predicates = tuple(predicates)
        return output_predicates


class FOLNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        dims = config.predicate_dims
        if hasattr(config, "diag_link"):
            self.diag_link = config.diag_link
        else:
            self.diag_link = False
        if config.mixer_ops[0] is None or len(config.mixer_ops[0]) == 0:
            self.glob_link = None
            self.glob_branch = False
        else:
            self.glob_link = nn.Linear(dims[0], dims[1])
            self.glob_branch = True
        self.encoder = Encoder(config)
        self.reasoner = Reasoner(config)
        self.dense = nn.Linear(dims[1], dims[1])
        self.activation = nn.Tanh()
        self.diag = nn.Linear(dims[2], dims[1]) if self.diag_link else None
        self.initializer_range = config.initializer_range
        self.tie_weights()
        self.init_weights()

    def tie_weights(self):
        if self.glob_branch:
            for layer in self.reasoner.layers:
                boolean_0 = layer.deducers[1].operators[0][0]
                boolean_1 = layer.deducers[1].operators[1][0]
                boolean_0.clause_in = boolean_1.clause_in
                boolean_0.clause_out = boolean_1.clause_out
            for layer in self.reasoner.layers:
                glob_s, glob_n = [], []
                for operators in layer.deducers[0].operators:
                    for op in operators:
                        if isinstance(op, GatherOperator):
                            glob_s.append(op)
                        elif isinstance(op, ScatterOperator):
                            glob_s.append(op)
                        elif isinstance(op, FuseOperator):
                            glob_s.append(op)
                        elif isinstance(op, ReflectionOperator):
                            glob_n.append(op)
                        elif isinstance(op, LinkOperator):
                            glob_n.append(op)
                for op in glob_s:
                    op.premise = glob_s[0].premise
                for op in glob_n:
                    op.kernel = glob_n[0].kernel
                    op.premise = glob_n[0].premise
                del glob_s, glob_n

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.apply(self._init_weights)

    def forward(self, tok_seq=None, token_type_ids=None):
        base_predicates = self.encoder(tok_seq, token_type_ids)
        output_predicates = self.reasoner(base_predicates)
        embd_rel = output_predicates[2]
        embd_seq = output_predicates[1]
        if self.diag_link:
            diag2 = torch.diagonal(embd_rel, dim1=1, dim2=2)
            diag2 = diag2.permute(0, 2, 1)
            diag2 = self.diag(diag2)
            embd_seq = embd_seq + diag2
        embd_pool = embd_seq[:, 0, :]
        embd_pool = self.activation(self.dense(embd_pool))
        return embd_seq, embd_pool, embd_rel


class PreTrainHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = BertPreTrainingHeads(config)
        self.initializer_range = config.initializer_range
        self.init_weights()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.apply(self._init_weights)

    def forward(self, embd_seq, embd_pool, embd_rel):
        mlm_prediction, nsp_prediction = self.heads(embd_seq, embd_pool)
        return mlm_prediction, nsp_prediction


class SequenceContrastivePreTrainHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.initializer_range = config.initializer_range
        self.init_weights()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.apply(self._init_weights)

    def forward(self, embd_pool):
        embd_pool = self.dense(embd_pool)
        return embd_pool


class FOLNetForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.folnet = FOLNet(config)
        self.cls = PreTrainHeads(config)
        self.vocab_size = config.vocab_size
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.pretrain_loss = config.pretrain_loss
        if "SCL" in self.pretrain_loss:
            self.sch = SequenceContrastivePreTrainHead(config)
        else:
            self.sch = None
        self.tie_weights()

    def get_input_embeddings(self):
        return self.folnet.encoder.word_embd

    def get_output_embeddings(self):
        return self.cls.heads.predictions.decoder

    def tie_weights(self):
        input_embds = self.get_input_embeddings()
        output_embds = self.get_output_embeddings()
        self._tie_weights(output_embds, input_embds)

    def _tie_weights(self, output_embds, input_embds):
        output_embds.weight = input_embds.weight
        if getattr(output_embds, "bias", None) is not None:
            pad = output_embds.weight.shape[0] - output_embds.bias.shape[0]
            output_embds.bias.data = F.pad(
                output_embds.bias.data, (0, pad,), "constant", 0
            )
        if hasattr(output_embds, "out_features") \
        and hasattr(input_embds, "num_embeddings"):
            output_embds.out_features = input_embds.num_embeddings

    def compute_sc_loss(self, embd_scl):
        # compute cosine similarity: 2B x D1 -> 2B x 2B
        embd_scl = F.normalize(embd_scl, dim=1)
        scl_logits = torch.matmul(embd_scl, embd_scl.transpose(0, 1))
        # create scl labels and mask
        scl_labels = torch.arange(embd_scl.shape[0], device=embd_scl.device)
        scl_labels = scl_labels.view(-1, 2).flip(dims=(1,)).view(-1)
        scl_mask = torch.zeros_like(scl_logits)
        scl_mask.fill_diagonal_(-10000.0)
        # compute scl_loss and scl_error
        scl_logits = scl_logits + scl_mask
        scl_loss = self.criterion(scl_logits, scl_labels)
        scl_preds = scl_logits.argmax(dim=1)
        scl_error = (scl_preds != scl_labels).float().mean()
        return scl_loss, scl_error

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        config=None
    ):
        if config is None:
            pretrained_config_path = pretrained_model_path + '.cfg'
            config = FOLNetConfig.from_pretrained(pretrained_config_path)
        model = cls(config)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(
            pretrained_model_path + '.pt',
            map_location=torch.device('cpu')
        )
        src_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(src_dict)
        model.load_state_dict(model_dict)
        s = model.state_dict()
        unused_keys = [k for k in pretrained_dict.keys() if k not in model_dict]
        uninit_keys = [k for k in model_dict.keys() if k not in pretrained_dict]
        failed_keys = [k for k,v in src_dict.items() if torch.norm(v-s[k])>1e-6]
        print("unused pretrained weights = {}".format(unused_keys))
        print("randomly initialized weights = {}".format(uninit_keys))
        print("unsuccessfully initialized weights = {}".format(failed_keys))
        return model

    def forward(
        self,
        tok_seq=None,
        token_type_ids=None,
        mlm_positions=None,
        mlm_labels=None,
        nsp_label=None,
    ):
        embd_seq, embd_pool, embd_rel = self.folnet(tok_seq, token_type_ids)
        if mlm_positions is not None:
            dim = embd_seq.shape[2]
            gather_index = mlm_positions.unsqueeze(-1).expand(-1, -1, dim)
            embd_seq = embd_seq.gather(1, gather_index)
        mlm_pred, nsp_pred = self.cls(embd_seq, embd_pool, embd_rel)
        if mlm_labels is not None and nsp_label is not None:
            mlm_pred_ = mlm_pred.view(-1, self.vocab_size)
            mlm_labels_ = mlm_labels.view(-1)
            mlm_loss = self.criterion(mlm_pred_, mlm_labels_)
            total_loss = mlm_loss
            mlm_mask = (mlm_labels_ != -1)
            mlm_error = (mlm_labels_ != mlm_pred_.argmax(dim=1)) * mlm_mask
            mlm_error = mlm_error.float().sum() / mlm_mask.float().sum()
            outputs = (
                mlm_error.unsqueeze(-1),
                mlm_pred
            )
            if "NSP" in self.pretrain_loss or "SOP" in self.pretrain_loss:
                nsp_pred_ = nsp_pred.view(-1, 2)
                nsp_label_ = nsp_label.view(-1)
                nsp_loss = self.criterion(nsp_pred_, nsp_label_)
                total_loss = total_loss + nsp_loss
                nsp_error = (nsp_label_ != nsp_pred_.argmax(dim=1))
                nsp_error = nsp_error.float().mean()
                outputs = outputs + (nsp_error.unsqueeze(-1), nsp_pred)
            if "SCL" in self.pretrain_loss:
                embd_scl = self.sch(embd_pool)
                scl_loss, scl_error = self.compute_sc_loss(embd_scl)
                total_loss = total_loss + scl_loss
                outputs = outputs + (scl_error.unsqueeze(-1), embd_scl)
            outputs = (total_loss.unsqueeze(-1),) + outputs
        elif mlm_labels is None and nsp_label is None:
            outputs = (mlm_pred, nsp_pred)
            if "SCL" in self.pretrain_loss:
                embd_scl = self.sch(embd_pool)
                outputs = outputs + (embd_scl,)
        else:
            raise NotImplementedError()
        return outputs


class FOLNetForSeqClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.num_classes
        self.folnet = FOLNet(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        if config.num_classes > 1: # classification
            self.output_mode = "classification"
            self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        elif config.num_classes == 1: # regression
            self.criterion = nn.MSELoss()
            self.output_mode = "regression"
        else:
            raise KeyError(config.num_classes)
        self.initializer_range = config.initializer_range
        self.init_weights()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.apply(self._init_weights)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path, # contains *.cfg (config) and *.pt (model)
        config=None # if None, *.cfg (config) of this class shall be in path
    ):
        if config is None:
            pretrained_config_path = pretrained_model_path + '.cfg'
            config = FOLNetConfig.from_pretrained(pretrained_config_path)
        model = cls(config)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(
            pretrained_model_path + '.pt',
            map_location=torch.device('cpu')
        )
        src_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(src_dict)
        model.load_state_dict(model_dict)
        s = model.state_dict()
        unused_keys = [k for k in pretrained_dict.keys() if k not in model_dict]
        uninit_keys = [k for k in model_dict.keys() if k not in pretrained_dict]
        failed_keys = [k for k,v in src_dict.items() if torch.norm(v.float()-s[k].float())>1e-6]
        print("unused pretrained weights = {}".format(unused_keys))
        print("randomly initialized weights = {}".format(uninit_keys))
        print("unsuccessfully initialized weights = {}".format(failed_keys))
        return model

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        labels=None
    ):
        outputs = self.folnet(input_ids, token_type_ids)
        embd_pool = outputs[1]
        embd_pool = self.dropout(embd_pool)
        prediction = self.classifier(embd_pool)
        outputs = (prediction,)
        if labels is not None:
            if self.output_mode == "classification":
                prediction_ = prediction.view(-1, self.num_classes)
                labels_ = labels.view(-1)
                loss = self.criterion(prediction_, labels_)
                error = (labels_ != prediction_.argmax(dim=1)).float().mean()
            elif self.output_mode == "regression":
                prediction_ = prediction.view(-1)
                labels_ = labels.view(-1)
                loss = self.criterion(prediction_, labels_)
                error = loss.detach()
            outputs = (loss.unsqueeze(-1), error.unsqueeze(-1)) + outputs
        return outputs


class FOLNetForMultipleChoice(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.num_classes
        self.folnet = FOLNet(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.initializer_range = config.initializer_range
        self.init_weights()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.apply(self._init_weights)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path, # contains *.cfg (config) and *.pt (model)
        config=None # if None, *.cfg (config) of this class shall be in path
    ):
        if config is None:
            pretrained_config_path = pretrained_model_path + '.cfg'
            config = FOLNetConfig.from_pretrained(pretrained_config_path)
        model = cls(config)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(
            pretrained_model_path + '.pt',
            map_location=torch.device('cpu')
        )
        src_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(src_dict)
        model.load_state_dict(model_dict)
        s = model.state_dict()
        unused_keys = [k for k in pretrained_dict.keys() if k not in model_dict]
        uninit_keys = [k for k in model_dict.keys() if k not in pretrained_dict]
        failed_keys = [k for k,v in src_dict.items() if torch.norm(v-s[k])>1e-6]
        print("unused pretrained weights = {}".format(unused_keys))
        print("randomly initialized weights = {}".format(uninit_keys))
        print("unsuccessfully initialized weights = {}".format(failed_keys))
        return model

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        labels=None
    ):
        batch_size = input_ids.shape[0]
        input_ids = input_ids.view(batch_size*self.num_classes, -1)
        token_type_ids = token_type_ids.view(batch_size*self.num_classes, -1)
        outputs = self.folnet(input_ids, token_type_ids)
        embd_pool = outputs[1]
        embd_pool = self.dropout(embd_pool)
        prediction = self.classifier(embd_pool)
        prediction = prediction.view(-1, self.num_classes)
        outputs = (prediction,)
        if labels is not None:
            prediction_ = prediction.view(-1, self.num_classes)
            labels_ = labels.view(-1)
            loss = self.criterion(prediction_, labels_)
            error = (labels_ != prediction_.argmax(dim=1)).float().mean()
            outputs = (loss.unsqueeze(-1), error.unsqueeze(-1)) + outputs
        return outputs


class FOLNetForZeroShotPrediction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.num_classes
        self.folnet = FOLNet(config)
        self.cls = PreTrainHeads(config)
        self.vocab_size = config.vocab_size
        self.tie_weights()

    def get_input_embeddings(self):
        return self.folnet.encoder.word_embd

    def get_output_embeddings(self):
        return self.cls.heads.predictions.decoder

    def tie_weights(self):
        input_embds = self.get_input_embeddings()
        output_embds = self.get_output_embeddings()
        self._tie_weights(output_embds, input_embds)

    def _tie_weights(self, output_embds, input_embds):
        output_embds.weight = input_embds.weight
        if getattr(output_embds, "bias", None) is not None:
            pad = output_embds.weight.shape[0] - output_embds.bias.shape[0]
            output_embds.bias.data = F.pad(
                output_embds.bias.data, (0, pad,), "constant", 0
            )
        if hasattr(output_embds, "out_features") \
        and hasattr(input_embds, "num_embeddings"):
            output_embds.out_features = input_embds.num_embeddings

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        config=None
    ):
        if config is None:
            pretrained_config_path = pretrained_model_path + '.cfg'
            config = FOLNetConfig.from_pretrained(pretrained_config_path)
        model = cls(config)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(
            pretrained_model_path + '.pt',
            map_location=torch.device('cpu')
        )
        src_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(src_dict)
        model.load_state_dict(model_dict)
        s = model.state_dict()
        unused_keys = [k for k in pretrained_dict.keys() if k not in model_dict]
        uninit_keys = [k for k in model_dict.keys() if k not in pretrained_dict]
        failed_keys = [k for k,v in src_dict.items() if torch.norm(v-s[k])>1e-6]
        print("unused pretrained weights = {}".format(unused_keys))
        print("randomly initialized weights = {}".format(uninit_keys))
        print("unsuccessfully initialized weights = {}".format(failed_keys))
        return model

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        mlm_positions=None,
        option_ids=None,
    ):
        embd_seq, embd_pool, embd_rel = self.folnet(input_ids, token_type_ids)
        if mlm_positions is not None:
            dim = embd_seq.shape[2]
            gather_index = mlm_positions.unsqueeze(-1).expand(-1, -1, dim)
            embd_seq = embd_seq.gather(1, gather_index)
        mlm_pred, nsp_pred = self.cls(embd_seq, embd_pool, embd_rel)
        mlm_pred = mlm_pred.squeeze(-2)
        prediction = mlm_pred.gather(1, option_ids)
        outputs = (prediction,)
        return outputs


class PoolerLogits(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, 1)
        self.dense.weight.data.normal_(mean=0.0, std=0.02)
        self.dense.bias.data.zero_()

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states).squeeze(-1)
        if p_mask is not None:
            x.masked_fill_(p_mask, -30000.0)
        return x


class SQuADHead(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.start_logits = PoolerLogits(hidden_size)
        self.end_logits = PoolerLogits(hidden_size)

    def forward(
        self,
        hidden_states,
        start_positions=None,
        end_positions=None,
        p_mask = None,
    ):
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)
        end_logits = self.end_logits(hidden_states, p_mask=p_mask)

        if start_positions is not None and end_positions is not None:
            def loss_fct(logits, targets):
                return F.nll_loss(
                    F.log_softmax(
                        logits.view(-1, logits.size(-1)),
                        dim=-1,
                        dtype=torch.float32,
                    ),
                    targets.view(-1),
                    reduction='sum',
                )
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) * 0.5
            return total_loss
        else:
            return start_logits, end_logits



class FOLNetForQuestionAnswering(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.num_classes
        self.folnet = FOLNet(config)
        self.classifier = SQuADHead(config.hidden_size)
        self.initializer_range = config.initializer_range
        self.init_weights()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.apply(self._init_weights)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path, # contains *.cfg (config) and *.pt (model)
        config=None # if None, *.cfg (config) of this class shall be in path
    ):
        if config is None:
            pretrained_config_path = pretrained_model_path + '.cfg'
            config = FOLNetConfig.from_pretrained(pretrained_config_path)
        model = cls(config)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(
            pretrained_model_path + '.pt',
            map_location=torch.device('cpu')
        )
        src_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(src_dict)
        model.load_state_dict(model_dict)
        s = model.state_dict()
        unused_keys = [k for k in pretrained_dict.keys() if k not in model_dict]
        uninit_keys = [k for k in model_dict.keys() if k not in pretrained_dict]
        failed_keys = [k for k,v in src_dict.items() if torch.norm(v.float()-s[k].float())>1e-6]
        print("unused pretrained weights = {}".format(unused_keys))
        print("randomly initialized weights = {}".format(uninit_keys))
        print("unsuccessfully initialized weights = {}".format(failed_keys))
        return model

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        start_positions=None,
        end_positions=None
    ):
        outputs = self.folnet(input_ids, token_type_ids)
        embd_seq = outputs[0]
        prediction = self.classifier(embd_seq, start_positions, end_positions)
        outputs = (prediction,)
        return outputs
