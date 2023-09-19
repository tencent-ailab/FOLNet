import os
import sys
import json
import argparse
import torch
from apex import amp


# s2b: parse input arguments into boolean variables, which will be used in the argparser.
def s2b(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean type expected.')


# s2l: parse input arguments into a list of boolean variables, which will be used in the argparser.
def s2l(s):
    return [int(item) for item in s.split(',')]


# s2sl: parse input arguments into a list of strings, which will be used in the argparser.
def s2sl(s):
    x = [item for item in s.split(',')]
    x = None if len(x) == 1 and x[0] == "None" else x
    return x


# save the finetuned model checkpoint, the performance results and the configuration file.
# It will only keep the most recent three checkpoints during the finetuning process.
# - Input arguments:
#     - net: the model
#     - output_path: the path for the output results and model checkpoints
#     - results: the performance metrics
#     - config: the configuration of the model
#     - tr_step: effective training steps
#     - saved_models: list of the currently saved model paths
#     - rank: the current rank
def save_finetuned_results(net, output_path, results, config, tr_step, saved_models, rank):
    if rank != 0:
        return
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


# Save the pretrained model checkpoints.
# - Input arguments:
#     - net: the model
#     - optimizer: the optimizer to be saved
#     - lr_scheduler: learning rate scheduler
#     - config: the configuration of the model
#     - epoch: the current number of epochs
#     - step: the total number of iteration steps (including the steps within
#       gradient accumulation)
#     - tr_step: the effective number of training update steps
#     - f_id: the id of the current training data shard
#     - args: the arguments
#     - output_path: the path for saving output model checkpoints
#     - saved_models: the currently saved models paths
#     - rank: the rank of the current process
def save_pretrained_results(
    net,
    optimizer,
    lr_scheduler,
    config,
    epoch,
    step,
    tr_step,
    f_id,
    args,
    output_path,
    saved_models,
    rank
):
    if rank != 0:
        return
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


# logging the training progress
# - Input arguments
#     - msg: the message to be logged
#     - output_path: the path for saving the log
#     - rank: the rank of the current process
def logging_results(msg, output_path, rank):
    sys.stdout.write(msg)
    sys.stdout.flush()
    log_file = output_path + '.rank' + str(rank) + ".log"
    with open(log_file, 'a') as log:
        log.write(msg)
        log.flush()


# add task / data arguments
def add_task_arguments(argparser):
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


# add arguments related to training parameters
def add_training_arguments(argparser):
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
    argparser.add_argument("--verbose_logging", action='store_true')


# add arguments related to model configurations
def add_model_arguments(argparser):
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


# Adding arguments that are specific to SQuAD task.
def add_squad_arguments(argparser):
    argparser.add_argument('--version_2_with_negative', action='store_true')
    argparser.add_argument('--null_score_diff_threshold', type=float, default=0.0)
    argparser.add_argument("--doc_stride", default=128, type=int)
    argparser.add_argument("--max_query_length", default=64, type=int)
    argparser.add_argument("--n_best_size", default=20, type=int)
    argparser.add_argument("--beam_size", default=20, type=int)
    argparser.add_argument("--max_answer_length", default=30, type=int)
