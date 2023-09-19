import random
import json
import argparse
import time
import os
import os.path
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (
    TensorDataset, DataLoader, SequentialSampler
)

from apex import amp

from folnet import FOLNetForQuestionAnswering as FOLNetForQA_PreLN_Full
from tokenization import BertTokenizer
from processors.utils_squad_evaluate import EvalOpts, main as evaluate_on_squad
from processors.utils_for_squad import (
    read_squad_examples,
    convert_examples_to_features,
    RawResult,
    write_predictions
)
from utils import save_finetuned_results as save_results
from utils import (
    add_task_arguments,
    add_training_arguments,
    add_model_arguments,
    add_squad_arguments
)
from utils import logging_results
from utils_opt import take_optimizer_step
from utils_opt import prep_finetune_optimizer_and_ddp as prep_optimizer_and_ddp
from utils_init import initialize_process, initialize_model
from utils_init import generate_finetune_model_info


MODEL_DICT = {
    "PreLN": FOLNetForQA_PreLN_Full,
    "PreLN.Full": FOLNetForQA_PreLN_Full,
}


MAIN_METRICS = {
    "squad": "best_f1",
}


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


# load cached data files if it exists, otherwise load data file and cache it.
# This will save the time for preprocessing the data again every time
def load_and_cache_examples(
    args,
    tokenizer,
    evaluate=False,
    output_examples=False
):
    # only the first process in distributed training process the dataset
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()
    # Load data features from cache or dataset file
    input_file_name = "dev-v2.0.json" if evaluate else "train-v2.0.json"
    input_file = os.path.join(args.data_dir, input_file_name)
    cached_features_file = os.path.join(
        os.path.dirname(input_file),
        'cached_{}_{}'.format(
            'dev' if evaluate else 'train',
            str(args.max_seq_length)
        )
    )
    if os.path.exists(cached_features_file):
        examples = read_squad_examples(
            input_file=input_file,
            is_training=not evaluate,
            version_2_with_negative=args.version_2_with_negative
        )
        print("Loading features from cached file ", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at ", input_file)
        examples = read_squad_examples(
            input_file=input_file,
            is_training=not evaluate,
            version_2_with_negative=args.version_2_with_negative
        )
        features = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            cls_token="[CLS]",  # tokenizer.cls_token,
            sep_token="[SEP]",  # tokenizer.sep_token,
            pad_token_id=args.pad_token,  # tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            cls_token_segment_id=0,
            pad_token_segment_id=0,
            cls_token_at_end=False,
            sequence_a_is_doc=False,
            add_two_separators=False
        )
        if args.local_rank in [-1, 0]:
            print("Saving features into cached file ", cached_features_file)
            torch.save(features, cached_features_file)
    # only the first process in distributed training process the dataset
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features],
        dtype=torch.long
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in features],
        dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features],
        dtype=torch.long
    )
    all_cls_index = torch.tensor(
        [f.cls_index for f in features],
        dtype=torch.long
    )
    all_p_mask = torch.tensor(
        [f.p_mask for f in features],
        dtype=torch.float
    )
    if evaluate:
        all_example_index = torch.arange(
            all_input_ids.size(0),
            dtype=torch.long
        )
        dataset = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_example_index,
            all_cls_index,
            all_p_mask
        )
    else:
        all_start_positions = torch.tensor(
            [f.start_position for f in features],
            dtype=torch.long
        )
        all_end_positions = torch.tensor(
            [f.end_position for f in features],
            dtype=torch.long
        )
        all_is_impossible = torch.tensor(
            [1 if f.is_impossible else 0 for f in features],
            dtype=torch.long
        )
        dataset = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_start_positions,
            all_end_positions,
            all_cls_index,
            all_p_mask,
            all_is_impossible
        )
    if output_examples:
        return dataset, examples, features
    return dataset


# evaluate the model on the dev set
# - Input arguments:
#   - model: the model to be evaluated
#   - args: the arguments
#   - tokenizer: the tokenizer for the data
# - Return:
#   - the performance metrics (dictionary)
def evaluate_model(model, args, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(
        args,
        tokenizer,
        evaluate=True,
        output_examples=True
    )
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset,
        sampler=eval_sampler,
        batch_size=args.batch_size_eval
    )
    # Evaluation
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = {}".format(len(dataset)))
    print("  Batch size = {}".format(args.batch_size_eval))
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.cuda(non_blocking=True) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0], 'token_type_ids': batch[2]}
            example_indices = batch[3]
            outputs = model(**inputs)
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(
                unique_id=unique_id,
                start_logits=to_list(outputs[0][0][i]),
                end_logits=to_list(outputs[0][1][i])
            )
            all_results.append(result)
    # Compute predictions
    output_prediction_file = os.path.join(
        args.output_dir,
        "predictions_{}.json".format(prefix)
    )
    output_nbest_file = os.path.join(
        args.output_dir,
        "nbest_predictions_{}.json".format(prefix)
    )
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(
            args.output_dir,
            "null_odds_{}.json".format(prefix)
        )
    else:
        output_null_log_odds_file = None
    tokens_to_text = None
    if hasattr(tokenizer, 'convert_tokens_to_string'):
        tokens_to_text = tokenizer.convert_tokens_to_string
    write_predictions(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokens_to_text=tokens_to_text
    )
    # Evaluate with the official SQuAD script
    evaluate_options = EvalOpts(
        data_file=os.path.join(args.data_dir, 'dev-v2.0.json'),
        pred_file=output_prediction_file,
        na_prob_file=output_null_log_odds_file
    )
    results = evaluate_on_squad(evaluate_options)
    metrics = {"detail": results}
    metrics["main"] = metrics["detail"][MAIN_METRICS[args.task_name]]
    os.remove(output_prediction_file)
    os.remove(output_nbest_file)
    os.remove(output_null_log_odds_file)
    return metrics


# generate the message for training information
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
    msg += "best=%.2f%% " % (results["metrics"]["main"])
    msg += "{"
    if results["metrics"]["detail"] is not None:
        for k, v in results["metrics"]["detail"].items():
            msg += k
            msg += "=%.2f " % v
        msg = msg[:-1]
    else:
        msg += "None"
    msg += "} "
    msg += "(%ds)\n" % (time_tot)
    return msg


# construct the dataloader for the training data
# - Input arguments:
#   - args: the arguments
#   - tokenizer: the tokenizer used for data
# - Return:
#   - the dataloader for training
def train_data_provider(args, tokenizer):
    train_data = load_and_cache_examples(
        args,
        tokenizer,
        evaluate=False,
        output_examples=False
    )
    tot_batch_size = args.batch_size * args.gradient_accumulation_steps
    args.max_steps = int(len(train_data)/tot_batch_size) * args.num_epochs
    print("number of samples = {}".format(len(train_data)))
    print("batch size = {}".format(tot_batch_size))
    print("number of steps = {}".format(args.max_steps))
    train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=int(args.batch_size/args.world_size)
    )
    return train_dataloader


# evaluate the model and save the metrics
# - Input arguments:
#   - args: the arguments
#   - net: the model
#   - config: the model configuration
#   - tokenizer: the tokenizer used for data
#   - step: the number of steps (including the steps between gradient
#   accumulation)
#   - tr_step: the actual number of model update steps
#   - rank: the rank of the current process
#   - model_description: the string that describes the model
#   - results: the performance results (dictionary)
#   - output_path: the path for saving the results
#   - saved_models: the paths to the most recent 3 models
def evaluate_and_save_results(
    args,
    net,
    config,
    tokenizer,
    step,
    tr_step,
    rank,
    model_description,
    results,
    output_path,
    saved_models
):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    eval_freq = args.eval_steps * args.gradient_accumulation_steps
    if (
        (step % eval_freq == 0 or tr_step >= args.max_steps)
        and rank == 0
    ):
        prefix = args.time_stamp + '.' + model_description
        prefix += '.rank' + str(rank) + '.step' + str(tr_step)
        metrics = evaluate_model(net, args, tokenizer, prefix)
        if metrics["main"] > results["metrics"]["main"]:
            results["step"] = tr_step
            results["metrics"] = metrics
            save_results(net, output_path, results, config, tr_step, saved_models, rank)
    if args.local_rank == 0:
        torch.distributed.barrier()


# this is the main training (finetuning) function
def train(args):
    # ==== initialize the training process ====
    rank, use_gpu = initialize_process(args)
    net, config = initialize_model(args, MODEL_DICT[args.model_type])
    if use_gpu:
        torch.cuda.set_device(args.local_rank)
        net.cuda(args.local_rank)
    tokenizer = BertTokenizer(
        args.vocab_file,
        do_lower_case=args.do_lower_case,
        max_len=512,
    )  # for bert large
    args.max_steps = None
    if args.do_train:
        train_dataloader = train_data_provider(args, tokenizer)
    net, optimizer, lr_scheduler = prep_optimizer_and_ddp(net, args)
    # ==== start finetuning the model ====
    print("================ Start finetuning ================")
    model_description = generate_finetune_model_info(args)
    output_path = args.output_dir + args.time_stamp + '.' + model_description
    overflow_buf = torch.cuda.IntTensor([0]) if args.allreduce_post else None
    results = {"step": 0, "train": 0.0, "metrics": {"main": 0.0, "detail": None}}
    step, tr_step = 0, 0
    train_loss_ = .0
    saved_models = []
    tic = time.time()
    while args.do_train:
        for _ in range(args.num_epochs):
            for bat in train_dataloader:
                step += 1
                # set the model to training mode
                net.train()
                # get the training data
                bat = tuple(t.cuda(non_blocking=True) for t in bat)
                inputs = {
                    'input_ids': bat[0],
                    'token_type_ids': bat[2],
                    'start_positions': bat[3],
                    'end_positions': bat[4]
                }
                # forward
                outputs = net(**inputs)
                # backward
                loss = outputs[0].mean()
                train_loss_ += loss.item()
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
                    train_loss_ = 0.0
                # evaluate the model on dev set
                evaluate_and_save_results(
                    args, net, config, tokenizer, step, tr_step, rank,
                    model_description, results, output_path, saved_models
                )
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


def init_args():
    argparser = argparse.ArgumentParser()
    add_task_arguments(argparser)
    add_squad_arguments(argparser)  # arguments specific to SQuAD task
    add_training_arguments(argparser)
    add_model_arguments(argparser)
    args = argparser.parse_args()
    args.task_name = args.task_name.lower()
    args.world_size = args.gpus_per_node * args.num_nodes
    if args.batch_size % args.world_size != 0:  # batch evenly spread over gpus
        msg = "batch_size should be divisible by %d" % (args.world_size)
        raise ValueError(msg)
    args.time_stamp = args.time_stamp.replace(' ', '-').replace(':', '-')
    return args


def main():
    # ==== input argument ====
    args = init_args()
    train(args)


if __name__ == "__main__":
    main()
