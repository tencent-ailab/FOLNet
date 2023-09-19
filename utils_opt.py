import torch
import amp_C
import apex_C
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel.distributed import flat_dist_call
from apex import amp
from apex.optimizers import FusedLAMB
from apex.optimizers import FusedAdam
from apex.amp import _amp_state
from schedulers import PolyWarmUpScheduler
from schedulers import LinearWarmUpScheduler


LR_SCHEDULER = {
    "poly": PolyWarmUpScheduler,
    "linear": LinearWarmUpScheduler
}


# This function prepares the optimizer and DistributedDataParallel (DDP) for
# pretraining. It can construct the optimizer and DDP from scratch, and it can
# also resume them from an existing checkpoint to continue pretraining.
# - Input arguments:
#   - net: the model
#   - args: the arguments
#   - checkpoint: the existing checkpoint
# - Return:
#   - net: the model that is ready for DDP
#   - optimizer: the pretraining optimizer
#   - lr_scheduler: the learning rate scheduler
def prep_pretrain_optimizer_and_ddp(net, args, checkpoint):
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


# This function prepares the optimizer and DistributedDataParallel (DDP) for
# finetuning. It constructs the optimizer and DDP from scratch.
# - Input arguments:
#   - net: the model
#   - args: the arguments
# - Return:
#   - net: the model that is ready for DDP
#   - optimizer: the pretraining optimizer
#   - lr_scheduler: the learning rate scheduler
def prep_finetune_optimizer_and_ddp(net, args):
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


# Takes optimizer step during training under the mixed precision setting
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
