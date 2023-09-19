import random
import json
import pickle
import argparse
import time
import os
import os.path
import numpy as np

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (
    TensorDataset, DataLoader
)

from apex import amp

from folnet import FOLNetForSeqClassification as FOLNetForSeqCls_PreLN_Full
from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr
from processors.glue import PROCESSORS, convert_examples_to_features
from utils import save_finetuned_results as save_results
from utils import add_task_arguments, add_training_arguments, add_model_arguments
from utils import logging_results
from utils_opt import take_optimizer_step
from utils_opt import prep_finetune_optimizer_and_ddp as prep_optimizer_and_ddp
from utils_init import initialize_process, initialize_model, initialize_data
from utils_init import generate_finetune_model_info


MODEL_DICT = {
    "PreLN": FOLNetForSeqCls_PreLN_Full,
}


# compute the performance metrics corresponding to each task.
# - Input arguments:
#   - task_name: the name of the specific task in GLUE
#   - preds: the prediction results
#   - labels: the groundtruth labels
# - Return:
#   - the corresponding metrics (in dictionary) of the corresponding task
def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


# the main metric for each task in GLUE
MAIN_METRICS = {
    "cola": "mcc",
    "sst-2": "acc",
    "mrpc": "acc_and_f1",
    "sts-b": "pearson_and_spearman",
    "qqp": "acc_and_f1",
    "mnli": "",
    "qnli": "acc",
    "rte": "acc",
    "wnli": "acc",
}


# standard accuracy
# - Input arguments:
#   - preds: the prediction results
#   - labels: the groundtruth labels
# - Return:
#   - the accuracy of the prediction
def simple_accuracy(preds, labels):
    return (preds == labels).mean()


# accuracy and F1 score
# - Input arguments
#   - preds: the prediction results
#   - labels: the groundtruth labels
# - Return:
#   - the accuracy and the F1 score (dictionary)
def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


# compute the accuracy from logit input
# - Input arguments:
#   - out: the prediction results in logits
#   - labels: the groundtruth labels
# - Return:
#   - the accuracy
def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


# pearson and spearman correlation coefficient (for STS-B task)
# - Input arguments:
#   - preds: the prediction results in logits
#   - labels: the groundtruth labels
# - Return:
#   - the Pearson and Spearman correlation coefficient (in dictionary)
def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "pearson_and_spearman": (pearson_corr + spearman_corr) / 2,
    }


# generate tensor dataset from the data features
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
        dtype=torch.float if type(features[0].label_id) == float else torch.long,
    )
    return TensorDataset(
        all_input_ids,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )


# generate the training features from text inputs.
# it will load cached file first if it exists. otherwise, load from data file
# and preprocess it. (also cache it to avoid preprocessing next time)
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
    except FileNotFoundError:
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


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


# evaluate the model performance on the dev set
# - Input arguments:
#   - net: the model
#   - args: the arguments
#   - tokenizer: the tokenizer used for the data
#   - processor: the task processor
#   - dev_type: the dev set split (the MNLI has two dev splits)
# - Return:
#   - the performance metric (dictionary)
def _evaluate_model(net, args, tokenizer, processor, dev_type=None):
    if dev_type is not None:
        eval_examples = processor.get_dev_examples(args.data_dir, dev_type)
    else:
        eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features, _ = convert_examples_to_features(
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
        for (_, bat) in enumerate(eval_dataloader):
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
    if len(processor.get_labels()) > 1:  # classification
        preds = np.argmax(preds, axis=1)
    elif len(processor.get_labels()) == 1:  # regression
        preds = preds.squeeze()
    eval_result = compute_metrics(args.task_name, preds, out_label_ids)
    metrics_names = [k for k in eval_result]
    metrics_values = [eval_result[k] for k in metrics_names]
    metrics_values = torch.Tensor(metrics_values).cuda(non_blocking=True)
    metrics_values /= args.world_size
    torch.distributed.all_reduce(metrics_values)
    metrics_values = metrics_values.detach().cpu().tolist()
    metrics = {k: v for k, v in zip(metrics_names, metrics_values)}
    return metrics


# this function is a wrapper for evaluation by considering the special cases
# related to MNLI, which has two dev splits
# - Input arguments:
#   - net: the model
#   - args: the arguments
#   - tokenizer: the tokenizer used for data
#   - processor: the task processor
# - Return:
#   - the performance metrics (dictionary)
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


# generate training information message
# - Input arguments:
#   - args: the arguments
#   - tr_step: the training steps
#   - train_loss: the training loss
#   - time_tot: the total amount of time used for training (finetuning)
#   - results: the performance results
# - Return:
#   - the training information message
def training_info(args, tr_step, train_loss, time_tot, results):
    msg = "step#%d/%d " % (tr_step, args.max_steps)
    msg += "Loss=%.2f " % (train_loss)
    msg += "best=%.2f%% " % (results["metrics"]["main"] * 100)
    msg += "{"
    if results["metrics"]["detail"] is not None:
        for k, v in results["metrics"]["detail"].items():
            msg += k
            msg += "=%.2f " % (v*100)
        msg = msg[:-1]
    else:
        msg += "None"
    msg += "} "
    msg += "(%ds)\n" % (time_tot)
    return msg


# this is the main training function (finetuning on GLUE)
def train(args):
    # ==== initialize the training process ====
    rank, use_gpu = initialize_process(args)
    net, config = initialize_model(args, MODEL_DICT[args.model_type])
    if use_gpu:
        torch.cuda.set_device(args.local_rank)
        net.cuda(args.local_rank)
    train_dataloader, processor, tokenizer = initialize_data(
        args, PROCESSORS[args.task_name], get_train_features, gen_tensor_dataset
    )
    net, optimizer, lr_scheduler = prep_optimizer_and_ddp(net, args)
    # ==== start finetuning the model ====
    print("================ Start finetuning ================")
    model_description = generate_finetune_model_info(args)
    output_path = args.output_dir + args.time_stamp + '.' + model_description
    overflow_buf = torch.cuda.IntTensor([0]) if args.allreduce_post else None
    results = {"step": 0, "train": 0.0, "metrics": {"main": 0.0, "detail": None}}
    step, tr_step = 0, 0
    train_loss_, train_loss2 = .0, .0
    train_error_ = 0.0
    saved_models = []  # track the most recent 3 models
    tic = time.time()
    while args.do_train:
        for _ in range(args.num_epochs):
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
                    args.gradient_accumulation_steps > 1
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
                        save_results(net, output_path, results, config, tr_step, saved_models, rank)
                # show progress
                time_tot = time.time() - tic
                log_freq = 10 * args.gradient_accumulation_steps
                if (step % log_freq == 0 or tr_step >= args.max_steps) and rank == 0:
                    msg = training_info(args, tr_step, train_loss, time_tot, results)
                    logging_results(msg, output_path, rank)
                # finish training if maximum steps are reached
                if tr_step >= args.max_steps:
                    return
    if args.do_train:
        print("total finetune time={}s".format(time_tot))
        print("best dev results={}".format(json.dumps(results, indent=1)))


def main():
    # ==== input argument ====
    argparser = argparse.ArgumentParser()
    add_task_arguments(argparser)
    add_training_arguments(argparser)
    add_model_arguments(argparser)
    args = argparser.parse_args()
    args.task_name = args.task_name.lower()
    args.world_size = args.gpus_per_node * args.num_nodes
    if args.batch_size % args.world_size != 0:  # batch evenly spread over gpus
        msg = "batch_size should be divisible by %d" % (args.world_size)
        raise ValueError(msg)
    args.time_stamp = args.time_stamp.replace(' ', '-').replace(':', '-')
    train(args)

if __name__ == "__main__":
    main()
