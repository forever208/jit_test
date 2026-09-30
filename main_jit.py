import argparse
import datetime
import numpy as np
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc

import copy
from engine_jit import train_one_epoch, evaluate

from denoiser import Denoiser


def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL', help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR', help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant', help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999, help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996, help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Starting epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str, help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int, help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float, help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float, help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float, help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int, help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=40, help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256, help='Generation batch size')
    parser.add_argument('--eval_ema', type=float, choices=(0.9999, 0.9996), default=0.9999, help='EMA decay whose weights are used for sampling',)
    parser.add_argument('--resume_epochs', type=int, nargs='+', default=None,
                        help='With --evaluate_gen: evaluate output_dir/checkpoint-{epoch}.pth for each epoch (overrides --resume)')
    parser.add_argument('--cfg_values', type=float, nargs='+', default=None,
                        help='With --evaluate_gen: sweep over these CFG scales (overrides --cfg)')
    parser.add_argument('--eval_ema_values', type=float, nargs='+', choices=(0.9999, 0.9996), default=None,
                        help='With --evaluate_gen: sweep over these EMA decays (overrides --eval_ema)')

    # dataset
    parser.add_argument('--data_path', default='./data/imagenet', type=str, help='Path to the dataset')
    parser.add_argument('--class_num', default=1000, type=int)

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir', help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='', help='Folder that contains checkpoint to resume from')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda', help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')

    return parser


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_checkpoint(checkpoint_path, model_without_ddp, optimizer=None):
    """Load model and EMA weights (and optionally optimizer state); return the checkpoint epoch or None."""
    # Training checkpoints include non-tensor metadata such as
    # argparse.Namespace, so they cannot use PyTorch 2.6+'s
    # weights_only=True default. Checkpoints passed via --resume are
    # expected to come from a trusted training run.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_without_ddp.load_state_dict(checkpoint['model'])

    ema_state_dict1 = checkpoint['model_ema1']
    ema_state_dict2 = checkpoint['model_ema2']
    model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
    model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
    print("Resumed checkpoint from", checkpoint_path)

    epoch = checkpoint.get('epoch')
    if optimizer is not None and 'optimizer' in checkpoint and epoch is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded optimizer & scaler state!")
    del checkpoint
    return epoch


def evaluate_sweep(model_without_ddp, args, checkpoint_paths, seed, log_writer=None):
    cfg_values = args.cfg_values or [args.cfg]
    ema_values = args.eval_ema_values or [args.eval_ema]
    num_runs = len(checkpoint_paths) * len(ema_values) * len(cfg_values)
    run_idx = 0
    for checkpoint_path in checkpoint_paths:
        epoch = load_checkpoint(checkpoint_path, model_without_ddp)
        start_epoch = epoch + 1 if epoch is not None else args.start_epoch
        for ema in ema_values:
            for cfg in cfg_values:
                run_idx += 1
                args.cfg = cfg
                args.eval_ema = ema
                model_without_ddp.cfg_scale = cfg
                print(
                    f"[{run_idx}/{num_runs}] Evaluating checkpoint {checkpoint_path} "
                    f"(epoch {start_epoch}) cfg={cfg:g} ema={ema:g}"
                )
                torch.cuda.empty_cache()
                with torch.random.fork_rng():
                    torch.manual_seed(seed)
                    with torch.no_grad():
                        fid, is_score = evaluate(model_without_ddp, args, start_epoch - 1, batch_size=args.gen_bsz, log_writer=log_writer)

                if misc.is_main_process():
                    with open(os.path.join(args.output_dir, 'FID.log'), 'a') as f:
                        print(
                            f'epoch={checkpoint_path} cfg={cfg:g} '
                            f'ema={ema:g} '
                            f'interval_min={args.interval_min:g} interval_max={args.interval_max:g} '
                            f'FID={fid:.5f} IS={is_score:.5f}',
                            file=f,
                        )
                    if log_writer is not None:
                        log_writer.flush()


def main(args):
    sweep_args = {
        '--resume_epochs': args.resume_epochs,
        '--cfg_values': args.cfg_values,
        '--eval_ema_values': args.eval_ema_values,
    }
    if not args.evaluate_gen:
        used = [name for name, value in sweep_args.items() if value]
        if used:
            raise ValueError(f'{", ".join(used)} can only be used with --evaluate_gen')

    eval_checkpoint_paths = []
    if args.evaluate_gen:
        if args.resume_epochs:
            eval_checkpoint_paths = [
                os.path.join(args.output_dir, f'checkpoint-{epoch}.pth')
                for epoch in args.resume_epochs
            ]
        elif args.resume:
            eval_checkpoint_paths = [args.resume]
        else:
            raise ValueError('--evaluate_gen requires --resume or --resume_epochs')
        missing = [path for path in eval_checkpoint_paths if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(f'Checkpoint not found: {", ".join(missing)}')

    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # Data augmentation transforms
    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor()
    ])

    dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed
    )
    print("Sampler_train =", sampler_train)

    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(seed)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=data_loader_generator,
        persistent_workers=True,  # fix the dataloader error
    )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Evaluate generation
    if args.evaluate_gen:
        evaluate_sweep(model_without_ddp, args, eval_checkpoint_paths, seed, log_writer=log_writer)
        return

    # Resume from checkpoint if provided
    checkpoint_path = args.resume if args.resume else None
    # checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        epoch = load_checkpoint(checkpoint_path, model_without_ddp, optimizer)
        if epoch is not None:
            args.start_epoch = epoch + 1
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)

        # Save checkpoint periodically
        if epoch % 5 == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        if epoch % args.eval_freq == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs) and epoch > 10:
            torch.cuda.empty_cache()
            cfg_scale = model_without_ddp.cfg_scale
            eval_ema = args.eval_ema
            try:
                # Keep the CFG evaluation and subsequent training RNG state
                # identical to the original single-evaluation behavior.
                model_without_ddp.cfg_scale = 1.0
                for ema in (args.ema_decay1, args.ema_decay2):
                    args.eval_ema = ema
                    with torch.random.fork_rng():
                        with torch.no_grad():
                            fid, is_score = evaluate(
                                model_without_ddp, args, epoch,
                                batch_size=args.gen_bsz, log_writer=log_writer
                            )

                    if misc.is_main_process():
                        with open(os.path.join(args.output_dir, 'eval_cond.log'), 'a') as f:
                            print(f'epoch={epoch} ema={ema:g} FID={fid:.5f} IS={is_score:.5f}', file=f)

                    torch.cuda.empty_cache()

                args.eval_ema = eval_ema
                model_without_ddp.cfg_scale = cfg_scale
                with torch.no_grad():
                    fid, is_score = evaluate(
                        model_without_ddp, args, epoch,
                        batch_size=args.gen_bsz, log_writer=log_writer
                    )

                if misc.is_main_process():
                    with open(os.path.join(args.output_dir, f'eval_cfg_{cfg_scale}.log'), 'a') as f:
                        print(f'epoch={epoch} FID={fid:.5f} IS={is_score:.5f}', file=f)
            finally:
                args.eval_ema = eval_ema
                model_without_ddp.cfg_scale = cfg_scale
                torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
