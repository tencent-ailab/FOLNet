# Input arguments:
#   - $1: gpu_ids;
#   - $2: step of pretrained model ckpt;
#   - $3: task name;
#   - $4: gpus_per_node;
#   - $5: port;
#   - $6: number of classes
#   - $7: number of epochs
#   - $8: training lr;
#   - $9: warmup ratio
#   - $10: training batch-size;
#   - $11: gradient accumulation steps;
#   - $12: evaluation batch-size;
#   - $13: weight decay;
#   - $14: evaluation frequency
#   - $15: saving frequency
#   - $16: seed
#   - $17: max_seq_length
# data information
export NCCL_IB_DISABLE=1
step=$2
task=${3:-"SQuAD"}
data_dir="./finetune_data/"${task}"/"
model_dir="" # use your custom directory name that contains the pretrained model checkpoint file
model_file="" # use your custom name of the pretrained model checkpoint file
vocab_file="./finetune_data/vocab.txt" # use your custom path to the vocabulary file
output_dir="./FOLNet/"${model_dir} # use your custom path to store the finetuned checkpoints
pretrained_model_path="./Experiments/FOLNet/"${model_dir}${model_file}".step"${step} # use your custome path to the pretrained checkpoint
max_seq_length=${17:-"128"} # For SQuAD we can use 384 or 512
doc_stride=128
# computation configuration
gpu_devices=$1
num_nodes=1
gpus_per_node=${4:-"4"} # 8; 1
node_rank=0
master_address="localhost" # local: "localhost"; multi-node: IP
master_port=${5:-"86420"}
# model configuration
model_type="PreLN"
max_position_offset=64
num_reasoner_layers=24
reasoner_dims="0,1024,64"
reasoner_hids="0,4096,256"
num_heads=16
head_size=64
mixer_ops0="None"
mixer_ops1="join,mu,cjoin"
mixer_ops2="assoc,trans,prod"
num_classes=${6:-"3"}
initializer_range=0.02
layer_norm_eps=1e-12
hidden_dropout_prob=0.1
# training hyper-parameters
num_epochs=${7:-"5"}
optimizer="FusedAdam" # current only choice
lr=${8:-"3e-5"}
beta1=0.9
beta2=0.999
warmup=${9:-"0.01"}
batch_size=${10:-"32"} # 6000-7000 for 8xGPU; 880 for 1xGPU
gradient_accumulation_steps=${11:-"1"}
batch_size_eval=${12:-"512"}
weight_decay=${13:-"0.01"}
allreduce_post=True
allreduce_post_fp16=True
max_grad_norm=-1
fp_opt_level="O2"
eval_steps=${14:-"1000"} # 1000
logging_steps=10 # 10
save_steps=${15:-"1000"} # 10000
from_pretrained=True
seed=${16:-"42"}
time_stamp=$(date)
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES=${gpu_devices} python3 -m torch.distributed.launch \
    --nproc_per_node=${gpus_per_node} \
    --nnode=${num_nodes} \
    --node_rank=${node_rank} \
    --master_addr=${master_address} \
    --master_port=${master_port} \
    finetune_squad.py \
        --task_name ${task} \
        --data_dir ${data_dir} \
        --output_dir ${output_dir} \
        --vocab_file ${vocab_file} \
        --max_seq_length ${max_seq_length} \
        --do_lower_case \
        --version_2_with_negative \
        --doc_stride ${doc_stride} \
        --num_nodes ${num_nodes} \
        --gpus_per_node ${gpus_per_node} \
        --node_rank ${node_rank} \
        --fp_opt_level ${fp_opt_level} \
        --optimizer ${optimizer} \
        --batch_size ${batch_size} \
        --batch_size_eval ${batch_size_eval} \
        --lr ${lr} \
        --warmup ${warmup} \
        --beta1 ${beta1} \
        --beta2 ${beta2} \
        --epsilon 1e-6 \
        --weight_decay ${weight_decay} \
        --max_grad_norm ${max_grad_norm} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --allreduce_post ${allreduce_post} \
        --allreduce_post_fp16 ${allreduce_post_fp16} \
        --num_epochs ${num_epochs} \
        --eval_steps ${eval_steps} \
        --logging_steps ${logging_steps} \
        --save_steps ${save_steps} \
        --do_train \
        --time_stamp "${time_stamp}" \
        --from_pretrained ${from_pretrained} \
        --pretrained_model_path ${pretrained_model_path} \
        --model_type ${model_type} \
        --num_heads ${num_heads} \
        --head_size ${head_size} \
        --mixer_ops0 ${mixer_ops0} \
        --mixer_ops1 ${mixer_ops1} \
        --mixer_ops2 ${mixer_ops2} \
        --hidden_dropout_prob ${hidden_dropout_prob} \
        --initializer_range ${initializer_range} \
        --layer_norm_eps ${layer_norm_eps} \
        --num_classes ${num_classes} \
        --max_position_offset ${max_position_offset} \
        --num_reasoner_layers ${num_reasoner_layers} \
        --reasoner_dims ${reasoner_dims} \
        --reasoner_hids ${reasoner_hids} \
        --seed ${seed}
