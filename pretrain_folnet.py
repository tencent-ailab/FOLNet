import random
import sys
import math
import json
import argparse
import time
import os
import os.path
from glob import glob
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler
from apex import amp
from dataset import Hdf5Dataset, DynDataset
from folnet import FOLNetForPreTraining as FOLNetForPreTraining_PreLN
from utils import s2b, s2l, s2sl, save_pretrained_results
from utils_opt import take_optimizer_step
from utils_opt import prep_pretrain_optimizer_and_ddp as prep_optimizer_and_ddp
from utils_init import initialize_process, initialize_pretraining_model
from utils_init import generate_pretrain_model_info


DATASET = {
    "hdf5": Hdf5Dataset,
    'dyn': DynDataset,
}


MODEL_DICT = {
    "PreLN": FOLNetForPreTraining_PreLN,
}


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


# construct the dataloader for pretraining
def data_provider(data_train, args, worker_init):
    if args.data_type == "hdf5":
        dataloader = DataLoader(
            data_train,
            batch_size=int(args.batch_size/args.world_size),
            pin_memory=True,
            num_workers=args.num_loaders,
            sampler=RandomSampler(data_train),
            worker_init_fn=worker_init
        )
    elif args.data_type == "dyn":
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
    return dataloader


# generate the training information message
def training_info(args, tr_step, train_loss, train_mlm, train_nsp, train_scl, time_tot):
    msg = "step#%d/%d " % (tr_step, args.max_steps)
    msg += "Loss=%.2f " % (train_loss)
    if "MLM" in args.pretrain_loss:
        msg += "MLM=%.2f%% " % (train_mlm*100)
    if "NSP" in args.pretrain_loss:
        msg += "NSP=%.2f%% " % (train_nsp*100)
    if "SOP" in args.pretrain_loss:
        msg += "SOP=%.2f%% " % (train_nsp*100)
    if "SCL" in args.pretrain_loss:
        msg += "SCL=%.2f%% " % (train_scl*100)
    msg += "(%ds)\n" % (time_tot)
    return msg


# update the training metrics, which are synchronized across workers
def update_metrics(args, train_loss_, train_mlm_, train_nsp_, train_scl_):
    metrics = [train_loss_, train_mlm_, train_nsp_, train_scl_]
    divisor = args.gradient_accumulation_steps * args.world_size
    metrics = [x/divisor for x in metrics]
    metrics = torch.Tensor(metrics).cuda(non_blocking=True)
    torch.distributed.all_reduce(metrics)
    train_loss, train_mlm, train_nsp, train_scl = [x for x in metrics.tolist()]
    return train_loss, train_mlm, train_nsp, train_scl


# update the pretraining loss
def update_loss(args, outputs, train_loss_, train_mlm_, train_nsp_, train_scl_):
    train_loss_ += outputs[0].mean().item()
    if "MLM" in args.pretrain_loss:
        train_mlm_ += outputs[1].mean().detach().item()
    if any(s in args.pretrain_loss for s in ("NSP", "SOP")):
        train_nsp_ += outputs[3].mean().detach().item()
    if "SCL" in args.pretrain_loss:
        train_scl_ += outputs[-2].mean().detach().item()
    return train_loss_, train_mlm_, train_nsp_, train_scl_


# set the training initial states from scratch or load from checkpoints
def set_training_state(args, checkpoint, ckpts, model_descr):
    if args.auto_resume and checkpoint is not None:
        output_path = args.output_dir + ckpts[0][:-5]  # use existing ckpt name
        step = checkpoint["step"]
        epoch = checkpoint["epoch"]
        f_id_start = checkpoint["f_id"]
        if "accumulation" in checkpoint:
            step /= checkpoint["accumulation"]
            step *= args.gradient_accumulation_steps
    else:
        output_path = args.output_dir + args.time_stamp + '.' + model_descr
        step, epoch, f_id_start = 0, 0, 0
    return output_path, step, epoch, f_id_start


# print the model configuration and the number of model parameters
def print_model_info(net, config):
    print("================ Model architecture ================")
    tot_params = sum(p.numel() for p in net.parameters())
    tot_er_params = sum(p.numel() for p in net.folnet.parameters())
    print("config={}".format(json.dumps(config.__dict__, indent=1)))
    print("tot_params={}".format(tot_params))
    print("tot_er_params={}".format(tot_er_params))


# the actual pretraining function
def train(args):
    # ==== initialize process ====
    rank, use_gpu = initialize_process(args)
    worker_init = WorkerInitObj(args.seed + rank*args.num_loaders)
    # ==== config the model, optimizer and distributed training =====
    net, config, checkpoint, ckpts = initialize_pretraining_model(
        args, MODEL_DICT[args.model_type], rank, use_gpu
    )
    net, optimizer, lr_scheduler = prep_optimizer_and_ddp(net, args, checkpoint)
    print_model_info(net, config)
    model_descr = generate_pretrain_model_info(args)
    # ==== start pretraining the model ====
    print("================ Start pretraining ================")
    overflow_buf = torch.cuda.IntTensor([0]) if args.allreduce_post else None
    output_path, step, epoch, f_id_start = set_training_state(args, checkpoint, ckpts, model_descr)
    tr_step = 0
    train_loss_, train_mlm_, train_nsp_, train_scl_, = .0, .0, .0, .0
    tic = time.time()
    saved_models = []  # track the most recent 3 models
    while True:
        # collect all the files recursively in the data folder
        files = [
            y for x in os.walk(args.data_path)
            for y in glob(os.path.join(x[0], '*training*.bin'))
        ]
        assert len(files) > 0
        print("#shards={}".format(len(files)))
        # make sure all the workers see the same file list
        files.sort()
        random.Random(args.seed + epoch).shuffle(files)
        f_id_end = int(math.ceil(len(files) / args.world_size))
        for f_id in range(f_id_start, f_id_end):
            data_file = files[(f_id * args.world_size + rank) % len(files)]
            data_train = DATASET[args.data_type](data_file, args)
            dataloader = data_provider(data_train, args, worker_init)
            for bat in iter(dataloader):
                step += 1
                # set the model to training mode
                net.train()
                # get the training data
                if use_gpu:
                    inputs = {
                        k: v.cuda(non_blocking=True).view(-1, v.shape[-1])
                        for k, v in bat.items()
                    }
                # forward
                outputs = net(**inputs)
                # backward
                loss = outputs[0].mean()
                train_loss_, train_mlm_, train_nsp_, train_scl_ = update_loss(
                    args, outputs, train_loss_, train_mlm_, train_nsp_, train_scl_
                )
                if (
                    args.gradient_accumulation_steps > 1
                    and not args.allreduce_post
                ):
                    loss = loss / args.gradient_accumulation_steps
                with amp.scale_loss(
                    loss, optimizer, delay_overflow_check=args.allreduce_post
                ) as scaled_loss:
                    scaled_loss.backward()
                del loss
                del outputs
                # update the model with gradient accumulation
                if step % args.gradient_accumulation_steps == 0:
                    lr_scheduler.step()
                    take_optimizer_step(args, optimizer, net, overflow_buf)
                    tr_step = int(step // args.gradient_accumulation_steps)
                    train_loss, train_mlm, train_nsp, train_scl = update_metrics(
                        args, train_loss_, train_mlm_, train_nsp_, train_scl_
                    )
                    train_loss_, train_mlm_, train_nsp_, train_scl_ = 0.0, 0.0, 0.0, 0.0
                # show progress
                time_tot = time.time() - tic
                log_freq = 10 * args.gradient_accumulation_steps
                if step % log_freq == 0 or tr_step >= args.max_steps:
                    if args.local_rank == 0:
                        msg = training_info(args, tr_step, train_loss, train_mlm, train_nsp, train_scl, time_tot)
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
                ):
                    save_pretrained_results(
                        net, optimizer, lr_scheduler, config, epoch, step,
                        tr_step, f_id, args, output_path, saved_models, rank
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
    argparser.add_argument('--max_span', type=int, default=64)  # depreciated
    argparser.add_argument('--span_dim', type=int, default=8)  # depreciated
    argparser.add_argument('--aux_length', type=int, default=64)  # depreciated
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
    if args.batch_size % args.world_size != 0:  # batch evenly spread over gpus
        msg = "batch_size should be divisible by %d" % (args.world_size)
        raise ValueError(msg)
    if "SCL" in args.pretrain_loss and args.data_type != "dyn":
        msg = "SCL loss must be trained on data_type=dyn"
        raise ValueError(msg)
    args.time_stamp = args.time_stamp.replace(' ', '-').replace(':', '-')
    train(args)

if __name__ == "__main__":
    main()
