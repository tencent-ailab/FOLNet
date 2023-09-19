import os
import random
import json
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
)
from config_folnet import FOLNetConfig
from tokenization import BertTokenizer


def initialize_process(args):
    use_gpu = (not args.disable_cuda) and torch.cuda.is_available()
    rank = args.gpus_per_node * args.node_rank + args.local_rank
    dist.init_process_group(
        backend = 'nccl' if use_gpu else 'gloo',
        init_method = 'env://',
    )
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    return rank, use_gpu


def initialize_model(args, model_class):
    print("================ Model architecture ================")
    if args.from_pretrained and args.pretrained_model_path is None:
        raise ValueError("pretrained_model_path should be given")
    if args.from_pretrained:
        pretrained_config_path = args.pretrained_model_path + '.cfg'
        config = FOLNetConfig.from_pretrained(pretrained_config_path)
        config.set_attr(num_classes=args.num_classes)
        net = model_class.from_pretrained(
            args.pretrained_model_path,
            config=config
        ).float()  # cast into FP32 in case it is FP16
    else:
        config = FOLNetConfig.from_args(args)
        config.set_attr(pretrain_loss=None)
        config.set_attr(num_classes=args.num_classes)
        net = model_class(config)
    print("config={}".format(json.dumps(config.__dict__, indent=1)))
    return net, config


def initialize_pretraining_model(args, model_class, rank, use_gpu):
    config = FOLNetConfig.from_args(args)
    if args.auto_resume and args.from_pretrained == "None":
        ckpts = [f for f in os.listdir(args.output_dir) if f.endswith(".ckpt")]
        if len(ckpts) == 0:  # no ckpt exists yet
            print("No ckpt found. Start from scratch.")
            net = model_class(config)
            checkpoint = None
        elif len(ckpts) == 1:  # one existing ckpt
            print("Resume from ckpt={}".format(ckpts[0]))
            net = model_class(config)
            checkpoint = torch.load(
                os.path.join(args.output_dir, ckpts[0]),
                map_location="cpu"
            )
            net.load_state_dict(checkpoint["model"])
        else:
            raise ValueError("More than one ckpt in the folder.")
    elif not args.auto_resume and args.from_pretrained == "None":
        print("Start from scratch.")
        net = model_class(config)
        ckpts = []
        checkpoint = None
    elif os.path.isfile(args.from_pretrained + ".pt"):
        print("Start from pretrained = {}".format(args.from_pretrained + ".pt"))
        if rank == 0:
            net = model_class.from_pretrained(
                args.from_pretrained,
                config=None if (
                    args.use_config_file
                    and
                    os.path.isfile(args.from_pretrained+".cfg")
                ) else config
            )
        else:
            net = model_class(config)
        ckpts = []
        checkpoint = None
    else:
        raise ValueError("Invalid args.auto_resume and args.from_pretrained.")
    if use_gpu:
        torch.cuda.set_device(args.local_rank)
        net.cuda(args.local_rank)
        print("Pretraining with GPU #{}".format(args.local_rank))
    else:
        print("Pretraining with CPU #{}".format(args.local_rank))
    return net, config, checkpoint, ckpts


def initialize_data(args, processor_class, get_train_features, gen_tensor_dataset):
    # ==== config the task processors ====
    processor = processor_class()
    tokenizer = BertTokenizer(
        args.vocab_file,
        do_lower_case=args.do_lower_case,
        max_len=512,
    )  # for bert large
    # ==== load dataset ====
    args.max_steps = None
    if args.do_train:
        train_feat = get_train_features(
            args.data_dir,
            args.max_seq_length,
            args.do_lower_case,
            tokenizer,
            processor,
            args.local_rank,
        )
        tot_batch_size = args.batch_size * args.gradient_accumulation_steps
        args.max_steps = int(len(train_feat)/tot_batch_size) * args.num_epochs
        print("number of samples = {}".format(len(train_feat)))
        print("batch size = {}".format(tot_batch_size))
        print("number of steps = {}".format(args.max_steps))
        train_data = gen_tensor_dataset(train_feat)
        train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=int(args.batch_size/args.world_size)
        )
    return train_dataloader, processor, tokenizer


def generate_finetune_model_info(args):
    return (
         'finetune'
         + '-' + args.task_name
         + '-' + 'FOLNet'
         + '-' + args.model_type
         + '-' + str(args.num_reasoner_layers) + 'L'
         + '-' + ".".join(map(str, args.reasoner_dims)) + 'H'
         + '-' + str(args.num_heads) + 'A'
         + '-' + '.' + args.optimizer
         + '.bsz' + str(args.batch_size*args.gradient_accumulation_steps)
         + '.lr' + str(args.lr)
         + '.ep' + str(args.num_epochs)
         + '.wu' + str(args.warmup)
         + '.seed' + str(args.seed)
    )


def generate_pretrain_model_info(args):
    return (
         'pretrain'
         + '-' + 'FOLNet'
         + '-' + args.model_type
         + '-' + args.pretrain_loss
         + '-' + str(args.num_reasoner_layers) + 'L'
         + '-' + ".".join(map(str, args.reasoner_dims)) + 'H'
         + '-' + str(args.num_heads) + 'A'
         + '-' + '.' + args.optimizer
         + '.bsz' + str(args.batch_size*args.gradient_accumulation_steps)
         + '.lr' + str(args.lr)
    )
