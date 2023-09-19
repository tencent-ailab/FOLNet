import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, RandomSampler
import numpy as np
import random
import sys
import math
import json
import argparse
import time
import os
import os.path
import amp_C
import apex_C
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel.distributed import flat_dist_call
from apex import amp
from apex.optimizers import FusedLAMB
from apex.amp import _amp_state
from datetime import datetime
from glob import glob
from dataset import Hdf5Dataset, BinDataset, DynDataset
from config_folnet import FOLNetConfig
from folnet import FOLNetForPreTraining as FOLNetForPreTraining_PreLN
from schedulers import PolyWarmUpScheduler
from schedulers import LinearWarmUpScheduler


DATASET = {
    "hdf5": Hdf5Dataset,
    "bin": BinDataset,
    'dyn': DynDataset,
}

MODEL_DICT = {
    "PreLN": FOLNetForPreTraining_PreLN,
}

LR_SCHEDULER = {
    "poly": PolyWarmUpScheduler,
    "linear": LinearWarmUpScheduler,
}


def s2b(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean type expected.')

def s2l(s):
    return [int(item) for item in s.split(',')]

def s2sl(s):
    x = [item for item in s.split(',')]
    x = None if len(x) == 1 and x[0] == "None" else x
    return x


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def prep_optimizer_and_ddp(net, args, checkpoint):
    # ==== config the optimizer ====
    param_optimizer = list(net.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': args.weight_decay
        },
        {
            'params': [
                p for n, p in param_optimizer
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        }
    ]
    if args.optimizer == "LAMB":
        optimizer = FusedLAMB(
            optimizer_grouped_parameters,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.epsilon,
        )
        lr_scheduler = LR_SCHEDULER[args.lr_scheduler](
            optimizer,
            warmup=args.warmup,
            total_steps=args.max_steps
        )
    else:
        raise NotImplementedError("Unknown optimizer")
    # ==== config distributed training ====
    if args.fp_opt_level in ('O0', 'O1', 'O2', 'O3'):
        net, optimizer = amp.initialize(
            net,
            optimizer,
            opt_level=args.fp_opt_level,
            loss_scale="dynamic",
            cast_model_outputs=torch.float16
        )
        init_loss_scale = 2**20
        scale_window = 200
        amp._amp_state.loss_scalers[0]._loss_scale = init_loss_scale
        amp._amp_state.loss_scalers[0]._scale_seq_len = scale_window
        if not args.allreduce_post:
            net = DDP(
                net,
                message_size=250000000,
                gradient_predivide_factor=args.world_size
            )
        else:
            flat_dist_call(
                [param.data for param in net.parameters()],
                torch.distributed.broadcast,
                (0,)
            )
    else:
        raise ValueError("Invalid fp_opt_level!")
    # ==== resume optimizer and lr_scheduler ====
    if args.auto_resume and checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if args.fp_opt_level in ('O0', 'O1', 'O2', 'O3'):
            optimizer._lazy_init_maybe_master_weights()
            optimizer._amp_stash.lazy_init_called = True
            optimizer.load_state_dict(checkpoint["optimizer"])
            amp_master = amp.master_params(optimizer)
            ckpt_master = checkpoint["master_params"]
            for p, s in zip(amp_master, ckpt_master):
                p.data.copy_(s.data)
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    return net, optimizer, lr_scheduler


def take_optimizer_step(args, optimizer, model, overflow_buf):
    if args.allreduce_post:
        # manually allreduce gradients after all accumulation steps
        # check for Inf/NaN
        # 1. allocate an uninitialized buffer for flattened gradient
        loss_scale = _amp_state.loss_scalers[0].loss_scale() \
            if args.fp_opt_level in ('O1', 'O2', 'O3') else 1
        master_grads = [
            p.grad for p in amp.master_params(optimizer) if p.grad is not None
        ]
        flat_grad_size = sum(p.numel() for p in master_grads)
        flat_raw = torch.empty(
            flat_grad_size,
            device='cuda',
            dtype=torch.float16 if args.allreduce_post_fp16 else torch.float32
        )
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(
            65536,
            overflow_buf,
            [master_grads, allreduced_views],
            loss_scale / (args.world_size * args.gradient_accumulation_steps)
        )
        # 3. sum gradient across ranks. Because of the predivision, this
        # averages the gradient
        torch.distributed.all_reduce(flat_raw)
        # 4. combine unscaling and unflattening of allreduced gradient
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(
            65536,
            overflow_buf,
            [allreduced_views, master_grads],
            1./loss_scale
        )
        # 5. update loss scale
        if args.fp_opt_level in ('O1', 'O2', 'O3'):
            scaler = _amp_state.loss_scalers[0]
            old_overflow_buf = scaler._overflow_buf
            scaler._overflow_buf = overflow_buf
            had_overflow = scaler.update_scale()
            scaler._overfloat_buf = old_overflow_buf
        else:
            had_overflow = 0
        # 6. call optimizer step function
        if had_overflow == 0:
            optimizer.step()
        else:
            # Overflow detected, print message and clear gradients
            if args.node_rank == 0:
                scaler = _amp_state.loss_scalers[0]
                print("loss_scale: {}".format(scaler.loss_scale()))
            if _amp_state.opt_properties.master_weights:
                for param in optimizer._amp_stash.all_fp32_from_fp16_params:
                    param.grad = None
        for param in model.parameters():
            param.grad = None
    else:
        optimizer.step()
        for param in model.parameters():
            param.grad = None
    return

def train(args):
    use_gpu = (not args.disable_cuda) and torch.cuda.is_available()
    # ==== initialize process ====
    rank = args.gpus_per_node * args.node_rank + args.local_rank
    dist.init_process_group(
        backend='nccl' if use_gpu else 'gloo',
        init_method='env://',
    )
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    worker_init = WorkerInitObj(args.seed + rank*args.num_loaders)
    # ==== config the model, optimizer and distributed training =====
    config = FOLNetConfig.from_args(args)
    if args.auto_resume and args.from_pretrained == "None":
        ckpts = [f for f in os.listdir(args.output_dir) if f.endswith(".ckpt")]
        if len(ckpts) == 0: # no ckpt exists yet
            print("No ckpt found. Start from scratch.")
            net = MODEL_DICT[args.model_type](config)
            checkpoint = None
        elif len(ckpts) == 1: # one existing ckpt
            print("Resume from ckpt={}".format(ckpts[0]))
            net = MODEL_DICT[args.model_type](config)
            checkpoint = torch.load(
                os.path.join(args.output_dir, ckpts[0]),
                map_location="cpu"
            )
            net.load_state_dict(checkpoint["model"])
        else:
            raise ValueError("More than one ckpt in the folder.")
    elif not args.auto_resume and args.from_pretrained == "None":
        print("Start from scratch.")
        net = MODEL_DICT[args.model_type](config)
        ckpts = []
        checkpoint = None
    elif os.path.isfile(args.from_pretrained + ".pt"):
        print("Start from pretrained = {}".format(args.from_pretrained + ".pt"))
        if rank == 0:
            net = MODEL_DICT[args.model_type].from_pretrained(
                args.from_pretrained,
                config=None if (
                    args.use_config_file
                    and
                    os.path.isfile(args.from_pretrained+".cfg")
                ) else config
            )
        else:
            net = MODEL_DICT[args.model_type](config)
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
    net, optimizer, lr_scheduler = prep_optimizer_and_ddp(net, args, checkpoint)
    print("================ Model architecture ================")
    tot_params = sum(p.numel() for p in net.parameters())
    tot_er_params = sum(p.numel() for p in net.folnet.parameters())
    print("config={}".format(json.dumps(config.__dict__, indent=1)))
    print("tot_params={}".format(tot_params))
    print("tot_er_params={}".format(tot_er_params))
    model_descr = (
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
    # ==== start pretraining the model ====
    print("================ Start pretraining ================")
    overflow_buf = torch.cuda.IntTensor([0]) if args.allreduce_post else None
    if args.auto_resume and checkpoint is not None:
        output_path = args.output_dir + ckpts[0][:-5] # use existing ckpt name
        step = checkpoint["step"]
        epoch = checkpoint["epoch"]
        f_id_start = checkpoint["f_id"]
        if "accumulation" in checkpoint:
            step /= checkpoint["accumulation"]
            step *= args.gradient_accumulation_steps
    else:
        output_path = args.output_dir + args.time_stamp + '.' + model_descr
        step = 0
        epoch = 0
        f_id_start = 0
    tr_step = 0
    train_loss_ = .0
    train_mlm_, train_nsp_, train_scl_, train_rtd_ = .0, .0, .0, .0
    tic = time.time()
    saved_models = []
    while True:
        files = [
            y for x in os.walk(args.data_path)
            for y in glob(os.path.join(x[0], '*training*.bin'))
        ]
        assert len(files) > 0
        print("#shards={}".format(len(files)))
        files.sort()
        random.Random(args.seed + epoch).shuffle(files)
        f_id_end = int(math.ceil(len(files) / args.world_size))
        for f_id in range(f_id_start, f_id_end):
            data_file = files[(f_id * args.world_size + rank) % len(files)]
            data_train = DATASET[args.data_type](data_file, args)
            if args.data_type == "hdf5":
                dataloader = DataLoader(
                    data_train,
                    batch_size=int(args.batch_size/args.world_size),
                    pin_memory=True,
                    num_workers=args.num_loaders,
                    sampler=RandomSampler(data_train),
                    worker_init_fn=worker_init
                )
            elif args.data_type == "bin" or "dyn":
                dataloader = DataLoader(
                    data_train,
                    batch_size=int(args.batch_size/args.world_size),
                    pin_memory=True,
                    num_workers=args.num_loaders,
                    sampler=None,
                    worker_init_fn=worker_init
                )
            else:
                raise ValueError("unknown data_type")
            for bat in iter(dataloader):
                step += 1
                # set the model to training mode
                net.train()
                # get the training data
                if use_gpu:
                    inputs = {
                        k:v.cuda(non_blocking=True).view(-1, v.shape[-1])
                        for k,v in bat.items()
                    }
                # forward
                outputs = net(**inputs)
                # backward
                loss = outputs[0].mean()
                train_loss_ += loss.item()
                if "MLM" in args.pretrain_loss:
                    train_mlm_ += outputs[1].mean().detach().item()
                if any(s in args.pretrain_loss for s in ("NSP", "SOP")):
                    train_nsp_ += outputs[3].mean().detach().item()
                if "RTD" in args.pretrain_loss:
                    train_rtd_ += outputs[1].mean().detach().item()
                if "CLM" in args.pretrain_loss:
                    train_rtd_ += outputs[1].mean().detach().item()
                    train_mlm_ += outputs[3].mean().detach().item()
                if "SCL" in args.pretrain_loss:
                    train_scl_ += outputs[-2].mean().detach().item()
                if (
                    args.gradient_accumulation_steps>1
                    and not args.allreduce_post
                ):
                    loss = loss / args.gradient_accumulation_steps
                with amp.scale_loss(
                    loss, optimizer, delay_overflow_check=args.allreduce_post
                ) as scaled_loss:
                    scaled_loss.backward()
                del loss
                del outputs
                if step % args.gradient_accumulation_steps == 0:
                    lr_scheduler.step()
                    take_optimizer_step(args, optimizer, net, overflow_buf)
                    tr_step = int(step // args.gradient_accumulation_steps)
                    train_loss = train_loss_/args.gradient_accumulation_steps
                    train_mlm = train_mlm_/args.gradient_accumulation_steps
                    train_nsp = train_nsp_/args.gradient_accumulation_steps
                    train_rtd = train_rtd_/args.gradient_accumulation_steps
                    train_scl = train_scl_/args.gradient_accumulation_steps
                    metrics = [train_loss, train_mlm, train_nsp, train_scl]
                    metrics = metrics + [train_rtd]
                    metrics = [x/args.world_size for x in metrics]
                    metrics = torch.Tensor(metrics).cuda(non_blocking=True)
                    torch.distributed.all_reduce(metrics)
                    train_loss = metrics[0].item()
                    train_mlm = metrics[1].item()
                    train_nsp = metrics[2].item()
                    train_scl = metrics[3].item()
                    train_rtd = metrics[4].item()
                    train_loss_ = 0.0
                    train_mlm_ = 0.0
                    train_nsp_ = 0.0
                    train_scl_ = 0.0
                    train_rtd_ = 0.0
                # show progress
                time_tot = time.time() - tic
                log_freq = 10 * args.gradient_accumulation_steps
                if step % log_freq == 0 or tr_step >= args.max_steps:
                    if args.local_rank == 0:
                        msg = "step#%d/%d " % (tr_step, args.max_steps)
                        msg += "Loss=%.2f " % (train_loss)
                        if "MLM" in args.pretrain_loss:
                            msg += "MLM=%.2f%% " % (train_mlm*100)
                        if "NSP" in args.pretrain_loss:
                            msg += "NSP=%.2f%% " % (train_nsp*100)
                        if "SOP" in args.pretrain_loss:
                            msg += "SOP=%.2f%% " % (train_nsp*100)
                        if "RTD" in args.pretrain_loss:
                            msg += "RTD=%.2f%% " % (train_rtd*100)
                        if "CLM" in args.pretrain_loss:
                            msg += "CLM=%.2f%% " % (train_mlm*100)
                            msg += "RTD=%.2f%% " % (train_rtd*100)
                        if "SCL" in args.pretrain_loss:
                            msg += "SCL=%.2f%% " % (train_scl*100)
                        msg += "(%ds)\n" % (time_tot)
                        sys.stdout.write(msg)
                        sys.stdout.flush()
                    if rank == 0:
                        with open(output_path+".log", 'a') as log:
                            log.write(msg)
                            log.flush()
                # save model
                if (
                    step%(args.save_freq*args.gradient_accumulation_steps) == 0
                    or tr_step >= args.max_steps
                ) and rank == 0:
                    net_ = net.module if hasattr(net, "module") else net
                    output_path_ = output_path + '.step' + str(tr_step)
                    torch.save(net_.state_dict(), output_path_ + '.pt')
                    with open(output_path_ + '.cfg', 'w') as f:
                        json.dump(config.__dict__, f, indent=1)
                    saved_models.append(output_path_)
                    if len(saved_models) > 3:
                        remove_model = saved_models.pop(0)
                        os.remove(remove_model + '.pt')
                        os.remove(remove_model + '.cfg')
                    torch.save(
                        {
                            "model": net_.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "master_params": list(amp.master_params(optimizer)),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "step": step,
                            "accumulation": args.gradient_accumulation_steps,
                            "f_id": f_id,
                        },
                        output_path + '.ckpt'
                    )
                # finish training if maximum steps are reached
                if tr_step >= args.max_steps:
                    return
            del dataloader
            del data_train
        f_id_start = 0
        epoch += 1
        worker_init.seed += 13577

def main():
    # ==== input argument ====
    argparser = argparse.ArgumentParser()
    # dataset hyper-parameters
    argparser.add_argument('--vocab_size', type=int, default=30522)
    argparser.add_argument('--type_vocab_size', type=int, default=2)
    argparser.add_argument('--data_path', type=str, required=True)
    argparser.add_argument('--data_type', type=str, default="bin")
    argparser.add_argument('--masked_lm_prob', type=float, default=0.15)
    argparser.add_argument('--max_predictions_per_seq', type=int, default=20)
    argparser.add_argument('--pad_token', type=int, default=0)
    argparser.add_argument('--cls_token', type=int, default=101)
    argparser.add_argument('--sep_token', type=int, default=102)
    argparser.add_argument('--mask_token', type=int, default=103)
    argparser.add_argument('--output_dir', type=str, required=True)
    argparser.add_argument('--num_loaders', type=int, default=0)
    # training hyper-parameters
    argparser.add_argument('--num_nodes', type=int, default=1)
    argparser.add_argument('--gpus_per_node', type=int, default=1)
    argparser.add_argument('--node_rank', type=int, default=0)
    argparser.add_argument('--local_rank', type=int)
    argparser.add_argument('--disable_cuda', action='store_true')
    argparser.add_argument('--fp_opt_level', type=str, default=None)
    argparser.add_argument('--optimizer', type=str, default="LAMB")
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--lr', type=float, default=1e-4)
    argparser.add_argument('--lr_scheduler', type=str, default="poly")
    argparser.add_argument('--warmup', type=float, default=0.01)
    argparser.add_argument('--beta1', type=float, default=0.9)
    argparser.add_argument('--beta2', type=float, default=0.999)
    argparser.add_argument('--epsilon', type=float, default=1e-6)
    argparser.add_argument('--weight_decay', type=float, default=0.1)
    argparser.add_argument('--max_grad_norm', type=float, default=-1)
    argparser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    argparser.add_argument('--allreduce_post', type=s2b, default=False)
    argparser.add_argument('--allreduce_post_fp16', type=s2b, default=True)
    argparser.add_argument('--max_steps', type=int, default=7038)
    argparser.add_argument('--verbose', type=int, default=0)
    argparser.add_argument('--save_freq', type=int, default=10000)
    argparser.add_argument('--save_to_all_nodes', type=s2b, default=False)
    argparser.add_argument('--use_config_file', type=s2b, default=True)
    argparser.add_argument('--from_pretrained', type=str, default="None")
    argparser.add_argument('--auto_resume', type=s2b, default=True)
    argparser.add_argument('--time_stamp', type=str, default="")
    argparser.add_argument('--seed', type=int, default=42)
    # model hyper-parameters
    argparser.add_argument('--model_type', type=str, default="PostLN")
    argparser.add_argument('--pretrain_loss', type=str, default="MLM.NSP")
    argparser.add_argument('--max_position_offset', type=int, default=128)
    argparser.add_argument('--absolute_position', type=s2b, default=False)
    argparser.add_argument('--relative_position', type=s2b, default=False)
    argparser.add_argument('--diag_link', type=s2b, default=False)
    argparser.add_argument('--num_reasoner_layers', type=int, default=12)
    argparser.add_argument('--reasoner_dims', type=s2l, default="0,768,64")
    argparser.add_argument('--reasoner_hids', type=s2l, default="0,3072,256")
    argparser.add_argument('--num_heads', type=int, default=12)
    argparser.add_argument('--head_size', type=int, default=64)
    argparser.add_argument('--max_span', type=int, default=64) # depreciated
    argparser.add_argument('--span_dim', type=int, default=8) # depreciated
    argparser.add_argument('--aux_length', type=int, default=64) # depreciated
    argparser.add_argument('--glob_size', type=int, default=384)
    argparser.add_argument('--span_size', type=int, default=48)
    argparser.add_argument('--unit_size', type=int, default=16)
    argparser.add_argument('--mixer_ops0', type=s2sl, default="None")
    argparser.add_argument('--mixer_ops1', type=s2sl, default="join")
    argparser.add_argument('--mixer_ops2', type=s2sl, default="assoc")
    argparser.add_argument('--boolean_type', type=str, default="mlp")
    argparser.add_argument('--hidden_act', type=str, default="gelu")
    argparser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    argparser.add_argument('--initializer_range', type=float, default=0.02)
    argparser.add_argument('--layer_norm_eps', type=float, default=1e-12)
    args = argparser.parse_args()
    args.world_size = args.gpus_per_node * args.num_nodes
    if args.batch_size % args.world_size != 0: # batch evenly spread over gpus
        msg = "batch_size should be divisible by %d" % (args.world_size)
        raise ValueError(msg)
    if "SCL" in args.pretrain_loss and args.data_type != "dyn":
        msg = "SCL loss must be trained on data_type=dyn"
        raise ValueError(msg)
    args.time_stamp = args.time_stamp.replace(' ','-').replace(':','-')
    train(args)

if __name__ == "__main__":
    main()
