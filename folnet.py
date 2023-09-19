import math
import numpy as np
from scipy.linalg import toeplitz

import torch
import torch.nn as nn
import torch.nn.functional as F
from config_folnet import FOLNetConfig


def _gelu_python(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

gelu = _gelu_python if torch.__version__ < "1.4.0" else F.gelu

ACT2FN = {
    "gelu": gelu,
    "relu": F.relu,
}


try:
    from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction
    from apex.normalization.fused_layer_norm import FusedLayerNormFunction
    APEX_IS_AVAILABLE = True and torch.cuda.is_available()
except ImportError:
    print("Better speed can be achieved with Nvidia apex package")
    APEX_IS_AVAILABLE = False


# initialization from pretrained model checkpoints
# - Input arguments:
#   - pretrained_model_path: the paths for pretrained model
#   - config: the configuration of the model
# - Return:
#   - the model loaded from pretrained checkpoint
def init_from_pretrained(cls, pretrained_model_path, config):
    if config is None:
        pretrained_config_path = pretrained_model_path + '.cfg'
        config = FOLNetConfig.from_pretrained(pretrained_config_path)
    model = cls(config)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(
        pretrained_model_path + '.pt',
        map_location=torch.device('cpu')
    )
    src_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(src_dict)
    model.load_state_dict(model_dict)
    s = model.state_dict()
    unused_keys = [k for k in pretrained_dict.keys() if k not in model_dict]
    uninit_keys = [k for k in model_dict.keys() if k not in pretrained_dict]
    failed_keys = [k for k, v in src_dict.items() if torch.norm(v.float() - s[k].float()) > 1e-6]
    print("unused pretrained weights = {}".format(unused_keys))
    print("randomly initialized weights = {}".format(uninit_keys))
    print("unsuccessfully initialized weights = {}".format(failed_keys))
    return model


# initialize model weights for certain types of modules
# - Input arguments
#   - initializer_range: the range of initialization (std)
def init_model_weights(module, initializer_range):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        if initializer_range < 0:
            init_std = math.sqrt(2.0/sum(list(module.weight.data.shape)))
        else:
            init_std = initializer_range
        module.weight.data.normal_(mean=0.0, std=init_std)
    elif isinstance(module, LayerNorm):
        if module.bias is not None:
            module.bias.data.zero_()
        if module.weight is not None:
            module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


# Layer Normalization
# This module is imported from apex that is fused version, and is more
# numerically stable for mixed precision training
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


# BertPredictionHeadTransform is a part of the operation / module for MLM head
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


# BertLMPredictionHead is the MLM head
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


# BertPreTrainingHeads is the module with MLM and NSP heads
class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


# The Encoder module is the part that takes tokens and other ids as the input
# and then output the corresponding embeddings.
# - Input arguments:
#   - tok_seq: the token id sequence
#   - tok_type_ids: the token type ids designating sequence #1 or #2 for NSP or
#   SOP loss settings
# - Output arguments:
#    - (embd0, embd1, embd2): the tuples that contains the embeddings for the
#    nullary, unary and binary predicates (the nullary one is always None)
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
        init_model_weights(module, self.initializer_range)

    def init_weights(self):
        self.apply(self._init_weights)

    def create_relative_position_ids(self, seq_length):
        # create relative position ids, which will be truncated if two tokens
        # are too far apart (beyond a certain threshold)
        r = np.arange(0, seq_length, 1)
        r = np.clip(r, None, self.max_position_offset-1)
        c = np.arange(0, -seq_length, -1)
        c = np.clip(c, -self.max_position_offset+1, None)
        c[1:] += 2 * self.max_position_offset
        rel_pos_ids = toeplitz(c, r)
        return rel_pos_ids

    def mask_relative_position_ids(self, rel_pos_ids, tok_type_ids):
        # mask the relative position ids (some of the relative positions are
        # not meaningful and will be masked out)
        batch_size = tok_type_ids.shape[0]
        dtype = tok_type_ids.dtype
        device = tok_type_ids.device
        rel_pos_ids = torch.tensor(rel_pos_ids, dtype=dtype, device=device)
        rel_pos_ids = rel_pos_ids.unsqueeze(0).expand(batch_size, -1, -1)
        rel_pos_mask = (tok_type_ids.unsqueeze(-1) == tok_type_ids.unsqueeze(-2))
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


# LogicOperator is the base class for all the remaining neural logic operator
# classes. It defines the forward function format and the child classes only
# need to define the _logic_op method.
# - Attributes:
#   - arity: the output arity of the operator (e.g., if the output is a binary
#   predicates, then the arity is 2)
#   - reader: the linear projections for the predicates of each arity
#   - writer: the linear projection that is applied to the output before
#   writing back into the residual streams
class LogicOperator(nn.Module):
    def __init__(self, config, arity):
        super().__init__()
        self.arity = arity  # output arity
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


# cjoin (c): U <- U x B
class ConjugateJoinOperator(nn.Module):  # based on new operator abstraction
    def __init__(self, config, arity):
        super().__init__()
        self.arity = arity  # output arity
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
        self.act_fun = nn.Softmax(dim=-1)  # Softmax/Identity
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # dropout/Identity

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


# join (j): U <- B x U
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


# mu (m): U <- B x B
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


# assoc (a): B <- U x U
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


# prod (p): B <- U x B
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


# trans (t): B <- B x B
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


# bool (b): U <- U x U & B <- B x B
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
            predicates_transp = predicates.transpose(1, 2)
            predicates = torch.cat((predicates, predicates_transp), dim=3)
            skip = self.clause_skip(predicates_transp)
        predicates = self.clause_in(predicates)
        predicates = self.act_fun(predicates)  # drop if ReGLU is used
        predicates = self.clause_out(predicates)
        if self.arity == 2:
            predicates = predicates + skip
        output_predicates = predicates
        return output_predicates


OPS = {
    "cjoin": ConjugateJoinOperator,
    "join": JoinOperator,
    "mu": MuOperator,
    "assoc": AssociativeOperator,
    "prod": ProductOperator,
    "trans": CompositionOperator,
    "bool": BooleanOperator,
}


# Put different neural logic operators into one module and make them interact
# (read and write) with the residual streams. FOLNet is a dual-branch
# architecture, which has two residual streams.
class Deduction(nn.Module):
    def __init__(self, config, deduction_type=None):
        super().__init__()
        predicate_dims = config.predicate_dims
        if config.mixer_ops[0] is None or len(config.mixer_ops[0]) == 0:
            ops_keys = {
                "object": config.mixer_ops,
                "permut": {0: None, 1: None, 2: ("perm",)},
                "predic": {0: None, 1: ("bool",), 2: ("bool",)}
            }[deduction_type]
        else:
            ops_keys = {
                "object": config.mixer_ops,
                "permut": {0: None, 1: None, 2: ("perm",)},
                "predic": {0: ("bool",), 1: ("bool",), 2: ("bool",)}
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


# Multiple Deduction stages are chained together into a ReasonerLayer
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


# Multiple ReasonerLayers are chained together into a Reasoner
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


# FOLNet is a cascading of Encoder and Reasoner, where the Encoder maps input
# ids and the relative position ides into embeddings and the Reasoner performs
# deduction process from basic predicates towards advanced predicates in the
# probabilistic logit space.
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
        assert not self.glob_branch

    def _init_weights(self, module):
        init_model_weights(module, self.initializer_range)

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
        init_model_weights(module, self.initializer_range)

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
        init_model_weights(module, self.initializer_range)

    def init_weights(self):
        self.apply(self._init_weights)

    def forward(self, embd_pool):
        embd_pool = self.dense(embd_pool)
        return embd_pool


# The class for pretraining with different settings of losses. The output
# linear layer is tied with the input embedding layer. It also allows for
# loading from pretrained checkpoints.
# - Input arguments:
#   - tok_seq: the input token ids
#   - token_type_ids: the token type ids indicating the sequence #1/#2
#   - mlm_positions: the positions for the masked tokens
#   - mlm_labels: the labels for MLM
#   - nsp_label: the labels for NSP (or SOP)
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
        return init_from_pretrained(cls, pretrained_model_path, config)

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


# The class for modeling FOLNet in the sequence classification setting. It can
# be used for classifying a single sequence or for classifying a pair of
# sequences. The embedding at [CLS] token will be used as the features for
# predicting the label. It can also be used in the regression setting as well.
class FOLNetForSeqClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.num_classes
        self.folnet = FOLNet(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        if config.num_classes > 1:  # classification
            self.output_mode = "classification"
            self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        elif config.num_classes == 1:  # regression
            self.criterion = nn.MSELoss()
            self.output_mode = "regression"
        else:
            raise KeyError(config.num_classes)
        self.initializer_range = config.initializer_range
        self.init_weights()

    def _init_weights(self, module):
        init_model_weights(module, self.initializer_range)

    def init_weights(self):
        self.apply(self._init_weights)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,  # contains *.cfg (config) and *.pt (model)
        config=None  # if None, *.cfg (config) of this class shall be in path
    ):
        return init_from_pretrained(cls, pretrained_model_path, config)

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


# The model class that uses FOLNet for multiple-choice tasks. There would be
# multiple streams, where each stream concatenates the passage, the question and
# one option. A linear projection will be applied to each stream and the stream
# with the highest score will be the predicted class.
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
        init_model_weights(module, self.initializer_range)

    def init_weights(self):
        self.apply(self._init_weights)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,  # contains *.cfg (config) and *.pt (model)
        config=None  # if None, *.cfg (config) of this class shall be in path
    ):
        return init_from_pretrained(cls, pretrained_model_path, config)

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


# The modeling class used for extractive question answering tasks. A linear
# classifier is applied to the final embedding at each token position to
# generate the prediction logits for each token position.
class FOLNetForQuestionAnswering(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.num_classes
        self.folnet = FOLNet(config)
        self.classifier = SQuADHead(config.hidden_size)
        self.initializer_range = config.initializer_range
        self.init_weights()

    def _init_weights(self, module):
        init_model_weights(module, self.initializer_range)

    def init_weights(self):
        self.apply(self._init_weights)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,  # contains *.cfg (config) and *.pt (model)
        config=None  # if None, *.cfg (config) of this class shall be in path
    ):
        return init_from_pretrained(cls, pretrained_model_path, config)

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
