"""Тренировочный цикл SAFMN с MLflow-логированием."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import structlog
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.data.dataset import RealEstateSRDataset
from training.data.degradation import DegradationConfig
from training.eval import evaluate, try_load_lpips
from training.gan import GANLoss, PatchDiscriminator
from training.losses import SRLoss
from training.model import SAFMN


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level.upper(), format="%(message)s", stream=sys.stderr)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.KeyValueRenderer(key_order=["event"]),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="safmn-train", description="Fine-tune SAFMN on HR photos")
    parser.add_argument("--hr-dir", type=Path, required=True, help="Папка с train HR-фото")
    parser.add_argument("--val-dir", type=Path, default=None, help="Папка с held-out HR для валидации")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--init-weights", type=Path, default=None, help="Стартовые веса (.pth)")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--crop", type=int, default=256, help="HR-кроп")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--perceptual-weight", type=float, default=0.1)
    parser.add_argument(
        "--scheduler", choices=["none", "cosine"], default="cosine", help="LR scheduler"
    )
    parser.add_argument("--use-gan", action="store_true", help="Включить PatchGAN adversarial loss")
    parser.add_argument("--gan-weight", type=float, default=0.1, help="Вес generator-loss в общем G-loss")
    parser.add_argument("--d-lr", type=float, default=1e-4, help="LR дискриминатора")
    parser.add_argument(
        "--lpips-eval", action="store_true", help="Считать LPIPS в валидации (нужен пакет lpips)"
    )
    parser.add_argument("--val-max-samples", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None, help="cpu/cuda; auto если не указано")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    parser.add_argument("--experiment", default="safmn-finetune")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _load_init_weights(model: torch.nn.Module, path: Path, log: structlog.BoundLogger) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("params", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    log.info("model.weights_loaded", path=str(path))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)
    log = structlog.get_logger("safmn_train")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = RealEstateSRDataset(
        hr_dir=args.hr_dir,
        hr_crop_size=args.crop,
        degradation=DegradationConfig(scale=args.scale),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    log.info("dataset.ready", samples=len(dataset), batches_per_epoch=len(loader))

    model = SAFMN(dim=128, n_blocks=16, ffn_scale=2.0, upscaling_factor=args.scale).to(device)
    if args.init_weights is not None:
        _load_init_weights(model, args.init_weights, log)

    criterion = SRLoss(perceptual_weight=args.perceptual_weight).to(device)
    optimizer_g = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler_g = (
        CosineAnnealingLR(optimizer_g, T_max=args.epochs) if args.scheduler == "cosine" else None
    )

    discriminator: PatchDiscriminator | None = None
    optimizer_d: torch.optim.Optimizer | None = None
    gan_loss: GANLoss | None = None
    if args.use_gan:
        discriminator = PatchDiscriminator().to(device)
        optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.d_lr, betas=(0.9, 0.99))
        gan_loss = GANLoss().to(device)
        log.info("gan.enabled", weight=args.gan_weight, d_lr=args.d_lr)

    lpips_fn = None
    if args.val_dir is not None and args.lpips_eval:
        lpips_fn = try_load_lpips(device)
        if lpips_fn is None:
            log.warning("lpips.not_installed", hint="pip install '.[lpips]'")

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "scale": args.scale,
                "crop": args.crop,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "perceptual_weight": args.perceptual_weight,
                "scheduler": args.scheduler,
                "use_gan": args.use_gan,
                "gan_weight": args.gan_weight,
                "device": str(device),
                "samples": len(dataset),
            }
        )

        global_step = 0
        best_psnr = -float("inf")
        for epoch in range(args.epochs):
            model.train()
            if discriminator is not None:
                discriminator.train()

            for lr_batch, hr_batch in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
                lr_batch = lr_batch.to(device, non_blocking=True)
                hr_batch = hr_batch.to(device, non_blocking=True)

                sr = model(lr_batch)
                content_loss, components = criterion(sr, hr_batch)
                g_loss = content_loss
                if discriminator is not None and gan_loss is not None:
                    d_fake_for_g = discriminator(sr)
                    g_adv = gan_loss.generator_loss(d_fake_for_g)
                    g_loss = content_loss + args.gan_weight * g_adv
                    components["g_adv"] = float(g_adv.item())
                    components["g_total"] = float(g_loss.item())

                optimizer_g.zero_grad(set_to_none=True)
                g_loss.backward()
                optimizer_g.step()

                if discriminator is not None and gan_loss is not None and optimizer_d is not None:
                    d_real = discriminator(hr_batch)
                    d_fake = discriminator(sr.detach())
                    d_loss = gan_loss.discriminator_loss(d_real, d_fake)
                    optimizer_d.zero_grad(set_to_none=True)
                    d_loss.backward()
                    optimizer_d.step()
                    components["d_loss"] = float(d_loss.item())

                if global_step % 10 == 0:
                    components["lr"] = float(optimizer_g.param_groups[0]["lr"])
                    mlflow.log_metrics(components, step=global_step)
                global_step += 1

            if scheduler_g is not None:
                scheduler_g.step()

            ckpt = args.out_dir / f"safmn_x{args.scale}_epoch{epoch + 1}.pth"
            torch.save({"params": model.state_dict()}, ckpt)
            mlflow.log_artifact(str(ckpt))
            log.info("epoch.done", epoch=epoch + 1, checkpoint=str(ckpt))

            if args.val_dir is not None:
                val_metrics = evaluate(
                    model=model,
                    val_dir=args.val_dir,
                    scale=args.scale,
                    device=device,
                    seed=epoch * 1000,
                    max_samples=args.val_max_samples,
                    lpips_fn=lpips_fn,
                )
                mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()}, step=global_step)
                log.info("val.done", epoch=epoch + 1, **val_metrics)

                if val_metrics["psnr"] > best_psnr:
                    best_psnr = val_metrics["psnr"]
                    best_path = args.out_dir / f"safmn_x{args.scale}_best.pth"
                    torch.save({"params": model.state_dict()}, best_path)
                    mlflow.log_artifact(str(best_path))
                    log.info("val.best", epoch=epoch + 1, psnr=best_psnr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
