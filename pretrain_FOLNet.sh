export NCCL_IB_DISABLE=1
# data information
seq=128
pred=20
data_path="" # your custom pretraining data path: all the files in the directory will be recursively collected and used in the pretraining
output_dir="" # your custom path for storing the model checkpoints
data_type="dyn" # 'idx', 'bin', 'dyn'
num_loaders=5
# computation configuration
gpu_devices=0,1,2,3,4,5,6,7
num_nodes=64
gpus_per_node=8
node_rank=$1 # the rank of the current node
master_address=$2 # the address of the master node
master_port=$3 # the port number
# model configuration
model_type="PreLN" # "PreLN" (="PreLN.FC"): Pre-LayerNorm architecture;
pretrain_loss="MLM.SOP" # MLM: Masked Language Model; NSP: Next sentence prediction; SOP: sequence order prediction;
max_position_offset=64
absolute_position=False
relative_position=True
diag_link=False
num_reasoner_layers=24
reasoner_dims="0,1024,64"
reasoner_hids="0,4096,256"
max_span=12
glob_size=64
num_heads=16
head_size=64
mixer_ops0="None"
mixer_ops1="join,mu,cjoin"
mixer_ops2="assoc,trans,prod"
boolean_type="mlp"
initializer_range=0.02
layer_norm_eps=1e-12
hidden_dropout_prob=0.1
# training hyper-parameters
max_steps=128000 # if max_steps is set, it will override num_epoch
optimizer="LAMB" # "BertAdam", "LAMB"
lr=1.6e-3
lr_scheduler="poly" # "linear", "poly"
beta1=0.9
beta2=0.999
warmup=0.01
weight_decay=0.01 # 0.01, 0.1
batch_size=2048
gradient_accumulation_steps=64
allreduce_post=True
allreduce_post_fp16=True
max_grad_norm=-1
fp_opt_level="O2"
from_pretrained="None" # "None": randinit; pretrained model path
auto_resume=True
use_config_file=False
save_freq=200
save_to_all_nodes=False # "True": save to all nodes; "False": to rank-0 node
verbose=0
time_stamp=$(date)
seed=89762 # can be any random seed
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES=${gpu_devices} python3 -m torch.distributed.launch \
    --nnode=${num_nodes} \
    --nproc_per_node=${gpus_per_node} \
    --node_rank=${node_rank} \
    --master_addr=${master_address} \
    --master_port=${master_port} \
    pretrain_folnet.py \
        --data_path ${data_path} \
        --output_dir ${output_dir} \
        --data_type ${data_type} \
        --masked_lm_prob 0.15 \
        --max_predictions_per_seq ${pred}\
        --num_loaders ${num_loaders} \
        --model_type ${model_type} \
        --pretrain_loss ${pretrain_loss} \
        --max_position_offset ${max_position_offset} \
        --absolute_position ${absolute_position} \
        --relative_position ${relative_position} \
        --diag_link ${diag_link} \
        --num_reasoner_layers ${num_reasoner_layers} \
        --reasoner_dims ${reasoner_dims} \
        --reasoner_hids ${reasoner_hids} \
        --max_span ${max_span} \
        --num_heads ${num_heads} \
        --head_size ${head_size} \
        --glob_size ${glob_size} \
        --mixer_ops0 ${mixer_ops0} \
        --mixer_ops1 ${mixer_ops1} \
        --mixer_ops2 ${mixer_ops2} \
        --boolean_type ${boolean_type} \
        --hidden_dropout_prob ${hidden_dropout_prob} \
        --layer_norm_eps ${layer_norm_eps} \
        --optimizer ${optimizer} \
        --batch_size ${batch_size} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --lr ${lr} \
        --lr_scheduler ${lr_scheduler} \
        --warmup ${warmup} \
        --beta1 ${beta1} \
        --beta2 ${beta2} \
        --epsilon 1e-6 \
        --weight_decay ${weight_decay} \
        --max_grad_norm ${max_grad_norm} \
        --from_pretrained ${from_pretrained} \
        --auto_resume ${auto_resume} \
        --initializer_range ${initializer_range} \
        --num_nodes ${num_nodes} \
        --gpus_per_node ${gpus_per_node} \
        --node_rank ${node_rank} \
        --fp_opt_level ${fp_opt_level} \
        --verbose ${verbose} \
        --max_steps ${max_steps} \
        --allreduce_post ${allreduce_post} \
        --allreduce_post_fp16 ${allreduce_post_fp16} \
        --save_freq ${save_freq} \
        --save_to_all_nodes ${save_to_all_nodes} \
        --use_config_file ${use_config_file} \
        --time_stamp "${time_stamp}" \
        --seed ${seed}
