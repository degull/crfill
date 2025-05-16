# logger.py (간이 버전 예시)
import os
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def add_scalar(self, tag, scalar_value, global_step):
        self.writer.add_scalar(tag, scalar_value, global_step)

    def add_single_image(self, tag, img, global_step):
        self.writer.add_image(tag, img, global_step)

    def add_single_label(self, tag, img, global_step):
        self.writer.add_image(tag, img, global_step)

    def write_console(self, epoch, iteration, time_per_iter):
        print(f"[Epoch {epoch}] Iter {iteration}, Time per iter: {time_per_iter:.4f}s")

    def write_scalar(self, tag, val, global_step):
        self.writer.add_scalar(tag, val, global_step)

    def write_html(self):
        # Optional: implement html logging (e.g. wandb, web view, etc.)
        pass
