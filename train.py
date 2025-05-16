import os
import torch
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
from data import create_dataloader  # 사용자 구현 데이터셋
from models.arrange_model import ArrangeModel
from util.iter_counter import IterationCounter
from logger.logger import Logger

# 1. 옵션 파싱 및 준비
opt = TrainOptions().parse()

# 2. 데이터 로더 생성
dataloader = create_dataloader(opt)

# 3. 모델 초기화
model = ArrangeModel(opt)

# 4. 옵티마이저 설정
optimizer_G, optimizer_D = model.create_optimizers(opt)

# 5. 반복자 및 로거 초기화
iter_counter = IterationCounter(opt, len(dataloader))
logger = Logger(log_dir=f'output/{opt.name}')

# ✅ 학습 시작 알림
print("🚀 Training started")

# 6. 에폭 반복
for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    print(f"\n🔄 Epoch [{epoch}/{opt.niter}]")

    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # 6.1. Discriminator 업데이트
        d_losses, _ = model(data_i, mode='discriminator')
        optimizer_D.zero_grad()
        d_loss_total = sum([v.mean() for v in d_losses.values()])
        d_loss_total.backward()
        optimizer_D.step()

        # 6.2. Generator 업데이트
        if i % opt.D_steps_per_G == 0:
            g_losses, _, _ = model(data_i, mode='generator')
            optimizer_G.zero_grad()
            g_loss_total = sum([v.mean() for v in g_losses.values()])
            g_loss_total.backward()
            optimizer_G.step()
        else:
            g_losses = {}

        # 6.3. 로그 출력 (터미널 출력 포함)
        if iter_counter.needs_displaying():
            for k, v in {**g_losses, **d_losses}.items():
                logger.add_scalar(k, v.mean().item(), iter_counter.total_steps_so_far)
            # 콘솔 출력
            log_str = f"[Epoch {epoch}/{opt.niter}] Iter {i}/{len(dataloader)} | " + \
                      " | ".join([f"{k}: {v.mean().item():.4f}" for k, v in {**g_losses, **d_losses}.items()])
            print(log_str)

    # 6.4. 에폭 종료 기록 및 모델 저장
    iter_counter.record_epoch_end()
    model.save(epoch)
    print(f"💾 Model saved at epoch {epoch}")

# python train.py --model arrange --netG baseconv --norm_type batch --use_th --th 0.5 --dataroot ./KADID10K/images --dataset_mode kadid