import json

class FOLNetConfig(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_args(cls, args):
        args.padded_vocab_size = (int(args.vocab_size // 8) + 1) * 8
        config_dict = {
            # common params
            "hidden_act": args.hidden_act,
            "hidden_dropout_prob": args.hidden_dropout_prob,
            "layer_norm_eps": args.layer_norm_eps,
            "initializer_range": args.initializer_range,
            # encoder params
            "max_position_offset": args.max_position_offset,
            "absolute_position": args.absolute_position,
            "relative_position": args.relative_position,
            "vocab_size": args.padded_vocab_size,
            "type_vocab_size": args.type_vocab_size,
            "diag_link": args.diag_link,
            # reasoner params
            "num_layers": args.num_reasoner_layers,
            "mixer_ops": {
                0: None if args.mixer_ops0 is None else tuple(args.mixer_ops0),
                1: tuple(args.mixer_ops1),
                2: tuple(args.mixer_ops2),
            },
            "boolean_type": args.boolean_type,
            "predicate_dims": tuple(args.reasoner_dims),
            "intermediate_dims": tuple(args.reasoner_hids),
            "num_heads": args.num_heads,
            "head_size": args.head_size,
            "max_span": args.max_span,
            "span_dim": args.span_dim,
            "aux_length": args.aux_length,
            "glob_size": args.glob_size,
            "span_size": args.span_size,
            "unit_size": args.unit_size,
            # output heads params
            "hidden_size": args.reasoner_dims[1],
            "pretrain_loss": args.pretrain_loss,
        }
        return cls(**config_dict)

    @classmethod
    def from_pretrained(cls, pretrained_config_path):
        with open(pretrained_config_path, 'r') as cfg_file:
            config = cls(**json.load(cfg_file))
        mixer_ops = {int(r):v for r, v in config.mixer_ops.items()}
        config.mixer_ops = mixer_ops
        return config

    def set_attr(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
