import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (
    TensorDataset, DataLoader, RandomSampler, SequentialSampler
)

import numpy as np
import random
import sys
import math
import json
import pickle
import argparse
import time
import os
import os.path
from datetime import datetime

import amp_C
import apex_C
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel.distributed import flat_dist_call
from apex import amp
from apex.optimizers import FusedLAMB
from apex.optimizers import FusedAdam
from apex.amp import _amp_state

from config_folnet import FOLNetConfig
from folnet import FOLNetForSeqClassification as FOLNetForSeqCls_PreLN_Full
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear
from schedulers import LinearWarmUpScheduler
from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr
from processors.folio import PROCESSORS, convert_examples_to_features


MODEL_DICT = {
    "PreLN": FOLNetForSeqCls_PreLN_Full,
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


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "folio":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

MAIN_METRICS = {
    "folio": "acc",
}


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "pearson_and_spearman": (pearson_corr + spearman_corr) / 2,
    }


def gen_tensor_dataset(features):
    all_input_ids = torch.tensor(
        [f.input_ids for f in features],
        dtype=torch.long,
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in features],
        dtype=torch.long,
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features],
        dtype=torch.long,
    )
    all_label_ids = torch.tensor(
        [f.label_id for f in features],
        dtype=torch.float if type(features[0].label_id)==float else torch.long,
    )
    return TensorDataset(
        all_input_ids,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )


def get_train_features(
    data_dir,
    max_seq_length,
    do_lower_case,
    tokenizer,
    processor,
    local_rank
):
    cache_file = os.path.join(
        data_dir,
        'cache_{0}_{1}'.format(
            str(max_seq_length),
            str(do_lower_case)
        )
    )
    train_features = None
    try:
        with open(cache_file, "rb") as reader:
            train_features = pickle.load(reader)
        print("Loaded pre-processed features from {}".format(cache_file))
    except:
        print("No pre-processed features from {}".format(cache_file))
        train_examples = processor.get_train_examples(data_dir)
        train_features, _ = convert_examples_to_features(
            train_examples,
            processor.get_labels(),
            max_seq_length,
            tokenizer,
        )
        if local_rank == 0:
            print("  Saving train feature into cache {}".format(cache_file))
            with open(cache_file, "wb") as writer:
                pickle.dump(train_features, writer)
    return train_features


def dump_predictions(path, label_map, preds, examples):
    label_rmap = {label_idx: label for label, label_idx in label_map.items()}
    predictions = {
        example.guid: label_rmap[preds[i]] for i, example in enumerate(examples)
    }
    with open(path, "w") as writer:
        json.dump(predictions, writer)


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def _evaluate_model(net, args, tokenizer, processor, dev_type=None):
    if dev_type is not None:
        eval_examples = processor.get_dev_examples(args.data_dir, dev_type)
    else:
        eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features, label_map = convert_examples_to_features(
        eval_examples,
        processor.get_labels(),
        args.max_seq_length,
        tokenizer
    )
    eval_data = gen_tensor_dataset(eval_features)
    eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data,
        sampler=eval_sampler,
        batch_size=int(args.batch_size_eval/args.world_size)
    )
    net.eval()
    preds = None
    out_label_ids = None
    with torch.no_grad():
        for (i_bat, bat) in enumerate(eval_dataloader):
            bat = tuple(t.cuda(non_blocking=True) for t in bat)
            input_ids, _, segment_ids, label_ids = bat
            logits = net(input_ids, token_type_ids=segment_ids)[0]
            logits_ = logits.detach().cpu().numpy()
            label_ids_ = label_ids.detach().cpu().numpy()
            if preds is None:
                preds, out_label_ids = logits_, label_ids_
            else:
                 preds = np.append(preds, logits_, axis=0)
                 out_label_ids = np.append(out_label_ids, label_ids_, axis=0)
    if len(processor.get_labels()) > 1: # classification
        preds = np.argmax(preds, axis=1)
    elif len(processor.get_labels()) == 1: # regression
        preds = preds.squeeze()
    eval_result = compute_metrics(args.task_name, preds, out_label_ids)
    metrics_names = [k for k in eval_result]
    metrics_values = [eval_result[k] for k in metrics_names]
    metrics_values = torch.Tensor(metrics_values).cuda(non_blocking=True)
    metrics_values /= args.world_size
    torch.distributed.all_reduce(metrics_values)
    metrics_values = metrics_values.detach().cpu().tolist()
    metrics = {k:v for k, v in zip(metrics_names, metrics_values)}
    return metrics


def evaluate_model(net, args, tokenizer, processor):
    if args.task_name == "mnli":
        m = _evaluate_model(net, args, tokenizer, processor, "dev_matched")
        mm = _evaluate_model(net, args, tokenizer, processor, "dev_mismatched")
        metrics = {"detail": {"m": m["acc"], "mm": mm["acc"]}}
        metrics["main"] = (m["acc"] + mm["acc"]) / 2.0
    else:
        metrics = {"detail": _evaluate_model(net, args, tokenizer, processor)}
        metrics["main"] = metrics["detail"][MAIN_METRICS[args.task_name]]
    return metrics


def prep_optimizer_and_ddp(net, args):
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
    optimizer, lr_scheduler = None, None
    if args.max_steps is not None:
        optimizer = FusedAdam(
            optimizer_grouped_parameters,
            lr=args.lr,
            bias_correction=False,
            betas=(args.beta1, args.beta2),
            eps=args.epsilon,
        )
    if args.fp_opt_level in ('O0', 'O1', 'O2', 'O3'):
        amp_inits = amp.initialize(
            net,
            optimizers=optimizer,
            opt_level=args.fp_opt_level,
            loss_scale="dynamic",
            keep_batchnorm_fp32=False,
        )
        init_loss_scale = 2**20
        scale_window = 200
        amp._amp_state.loss_scalers[0]._loss_scale = init_loss_scale
        amp._amp_state.loss_scalers[0]._scale_seq_len = scale_window
        net, optimizer = (
            amp_inits if args.max_steps is not None else (amp_inits, None)
        )
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
    if args.max_steps is not None:
        lr_scheduler = LinearWarmUpScheduler(
            optimizer,
            warmup=args.warmup,
            total_steps=args.max_steps
        )
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
    # ==== config and initialize the model =====
    print("================ Model architecture ================")
    if args.from_pretrained and args.pretrained_model_path is None:
        raise ValueError("pretrained_model_path should be given")
    if args.from_pretrained:
        pretrained_config_path = args.pretrained_model_path + '.cfg'
        config = FOLNetConfig.from_pretrained(pretrained_config_path)
        config.set_attr(num_classes=args.num_classes)
        net = MODEL_DICT[args.model_type].from_pretrained(
            args.pretrained_model_path,
            config=config
        ).float() # cast into FP32 in case it is FP16
    else:
        config = FOLNetConfig.from_args(args)
        config.set_attr(pretrain_loss=None)
        config.set_attr(num_classes=args.num_classes)
        net = MODEL_DICT[args.model_type](config)
    print("config={}".format(json.dumps(config.__dict__, indent=1)))
    if use_gpu:
        torch.cuda.set_device(args.local_rank)
        net.cuda(args.local_rank)
    # ==== config the task processors ====
    processor = PROCESSORS[args.task_name]()
    num_labels = len(processor.get_labels())
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
    # ==== config the optimizer, scheduler, AMP and DDP ====
    net, optimizer, lr_scheduler = prep_optimizer_and_ddp(net, args)
    # ==== start finetuning the model ====
    print("================ Start finetuning ================")
    if use_gpu:
        print("Finetuning with GPU #{}".format(args.local_rank))
    else:
        print("Finetuning with CPU #{}".format(args.local_rank))
    model_description = (
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
    output_path = args.output_dir + args.time_stamp + '.' + model_description
    overflow_buf = torch.cuda.IntTensor([0]) if args.allreduce_post else None
    results = {"step":0, "train":0.0, "metrics":{"main":0.0, "detail":None}}
    step, tr_step = 0, 0
    train_loss_, train_loss2 = .0, .0
    train_error_ = 0.0
    saved_models = []
    tic = time.time()
    while args.do_train:
        for epoch in range(args.num_epochs):
            for bat in train_dataloader:
                step += 1
                # set the model to training mode
                net.train()
                # get the training data
                bat = tuple(t.cuda(non_blocking=True) for t in bat)
                input_ids, _, segment_ids, label_ids = bat
                # forward
                outputs = net(
                    input_ids,
                    token_type_ids=segment_ids,
                    labels=label_ids
                )
                # backward
                loss = outputs[0].mean()
                error = outputs[1].mean()
                train_loss_ += loss.item()
                train_error_ += error.item()
                if (
                    args.gradient_accumulation_steps>1
                    and not args.allreduce_post
                ):
                    loss = loss / args.gradient_accumulation_steps
                with amp.scale_loss(
                    loss, optimizer, delay_overflow_check=args.allreduce_post
                ) as scaled_loss:
                    scaled_loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    lr_scheduler.step()
                    take_optimizer_step(args, optimizer, net, overflow_buf)
                    tr_step = int(step // args.gradient_accumulation_steps)
                    train_loss = train_loss_/args.gradient_accumulation_steps
                    train_loss2 += (train_loss - train_loss2) / tr_step
                    train_error = train_error_/args.gradient_accumulation_steps
                    train_loss_, train_error_ = 0.0, 0.0
                # evaluate the model on dev set
                eval_freq = args.eval_steps * args.gradient_accumulation_steps
                if step % eval_freq == 0 or tr_step >= args.max_steps:
                    metrics = evaluate_model(net, args, tokenizer, processor)
                    if metrics["main"] > results["metrics"]["main"]:
                        results["step"] = tr_step
                        results["train"] = 100 * (1 - train_error)
                        results["metrics"] = metrics
                        if rank == 0:
                            with open(output_path + ".eval", 'w') as f:
                                json.dump(results, f, indent=1)
                            net_ = net.module if hasattr(net, "module") else net
                            output_path_ = output_path + '.step' + str(tr_step)
                            torch.save(net_.state_dict(), output_path_ + '.pt')
                            with open(output_path_ + '.cfg', 'w') as f:
                                json.dump(config.__dict__, f, indent=1)
                            saved_models.append(output_path_)
                            if len(saved_models) > 1:
                                remove_model = saved_models.pop(0)
                                os.remove(remove_model + '.pt')
                                os.remove(remove_model + '.cfg')
                # show progress
                time_tot = time.time() - tic
                log_freq = 10 * args.gradient_accumulation_steps
                if (step%log_freq==0 or tr_step>=args.max_steps) and rank==0:
                    msg = "step#%d/%d " % (tr_step, args.max_steps)
                    msg += "Loss=%.2f " % (train_loss)
                    msg += "Acc=%.2f%% " % ((1 - train_error) * 100)
                    msg += "best=%.2f%% " % (results["metrics"]["main"] * 100)
                    msg += "{"
                    if results["metrics"]["detail"] is not None:
                        for k,v in results["metrics"]["detail"].items():
                            msg += k
                            msg += "=%.2f " % (v*100)
                        msg = msg[:-1]
                    else:
                        msg += "None"
                    msg += "} "
                    msg += "(%ds)\n" % (time_tot)
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    log_file = output_path + '.rank' + str(rank) + ".log"
                    with open(log_file, 'a') as log:
                        log.write(msg)
                        log.flush()
                # finish training if maximum steps are reached
                if tr_step >= args.max_steps:
                    return
    if args.do_train:
        print("total finetune time={}s".format(time_tot))
        print("best dev results={}".format(json.dumps(results, indent=1)))


def main():
    # ==== input argument ====
    argparser = argparse.ArgumentParser()
    # dataset hyper-parameters
    argparser.add_argument('--task_name', type=str, required=True)
    argparser.add_argument('--data_dir', type=str, required=True)
    argparser.add_argument('--output_dir', type=str, required=True)
    argparser.add_argument('--vocab_file', type=str, required=True)
    argparser.add_argument('--type_vocab_size', type=int, default=2)
    argparser.add_argument('--max_seq_length', type=int, default=128)
    argparser.add_argument('--cls_token', type=int, default=101)
    argparser.add_argument('--sep_token', type=int, default=102)
    argparser.add_argument('--pad_token', type=int, default=0)
    argparser.add_argument('--vocab_size', type=int, default=30522)
    argparser.add_argument('--do_lower_case', action='store_true')
    # training hyper-parameters
    argparser.add_argument('--num_nodes', type=int, default=1)
    argparser.add_argument('--gpus_per_node', type=int, default=1)
    argparser.add_argument('--node_rank', type=int, default=0)
    argparser.add_argument('--local_rank', type=int)
    argparser.add_argument('--disable_cuda', action='store_true')
    argparser.add_argument('--fp_opt_level', type=str, default=None)
    argparser.add_argument('--optimizer', type=str, default="FusedAdam")
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--batch_size_eval', type=int, default=256)
    argparser.add_argument('--lr', type=float, default=5e-3)
    argparser.add_argument('--warmup', type=float, default=0.01)
    argparser.add_argument('--beta1', type=float, default=0.9)
    argparser.add_argument('--beta2', type=float, default=0.999)
    argparser.add_argument('--epsilon', type=float, default=1e-6)
    argparser.add_argument('--weight_decay', type=float, default=0.01)
    argparser.add_argument('--max_grad_norm', type=float, default=-1)
    argparser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    argparser.add_argument('--allreduce_post', type=s2b, default=True)
    argparser.add_argument('--allreduce_post_fp16', type=s2b, default=True)
    argparser.add_argument('--num_epochs', type=int, default=5)
    argparser.add_argument('--eval_steps', type=int, default=1000)
    argparser.add_argument('--logging_steps', type=int, default=10)
    argparser.add_argument('--save_steps', type=int, default=10000)
    argparser.add_argument('--do_train', action='store_true')
    argparser.add_argument('--do_predict', action='store_true')
    argparser.add_argument('--from_pretrained', type=s2b, default=True)
    argparser.add_argument('--pretrained_model_path', type=str, default=None)
    argparser.add_argument('--time_stamp', type=str, default="")
    # model hyper-parameters
    argparser.add_argument('--model_type', type=str, default="PreLN")
    argparser.add_argument('--max_position_offset', type=int, default=128)
    argparser.add_argument('--num_reasoner_layers', type=int, default=12)
    argparser.add_argument('--reasoner_dims', type=s2l, default="768,768,32")
    argparser.add_argument('--reasoner_hids', type=s2l, default="3072,3072,128")
    argparser.add_argument('--num_heads', type=int, default=12)
    argparser.add_argument('--head_size', type=int, default=64)
    argparser.add_argument('--max_span', type=int, default=64)
    argparser.add_argument('--span_dim', type=int, default=8)
    argparser.add_argument('--mixer_ops0', type=s2sl, default="None")
    argparser.add_argument('--mixer_ops1', type=s2sl, default="self,join")
    argparser.add_argument('--mixer_ops2', type=s2sl, default="self,assoc")
    argparser.add_argument('--hidden_act', type=str, default="gelu")
    argparser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    argparser.add_argument('--initializer_range', type=float, default=0.02)
    argparser.add_argument('--layer_norm_eps', type=float, default=1e-12)
    argparser.add_argument('--num_classes', type=int, default=3)
    argparser.add_argument('--seed', type=int, default=42)
    args = argparser.parse_args()
    args.task_name = args.task_name.lower()
    args.world_size = args.gpus_per_node * args.num_nodes
    if args.batch_size % args.world_size != 0: # batch evenly spread over gpus
        msg = "batch_size should be divisible by %d" % (args.world_size)
        raise ValueError(msg)
    args.time_stamp = args.time_stamp.replace(' ','-').replace(':','-')
    train(args)

if __name__ == "__main__":
    main()
