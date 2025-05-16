import os
import pandas as pd  # 누락된 경우 추가
import random
from PIL import Image
import numpy as np
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from models.create_mask import MaskCreator

class KadidDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(preprocess_mode='scale_width_and_crop', load_size=512, crop_size=512, display_winsize=512)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # ✅ 수정된 경로: dataroot 상위 폴더에서 CSV 불러오기
        csv_path = os.path.join(os.path.dirname(self.root), 'kadid10k.csv')
        self.df = pd.read_csv(csv_path)
        
        # ✅ distortion 이미지들만 사용
        self.image_paths = [os.path.join(self.root, row['dist_img']) for _, row in self.df.iterrows()]
        self.mask_creator = MaskCreator()
        self.dataset_size = len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        params = get_params(self.opt, image.size)
        transform = get_transform(self.opt, params)
        image_tensor = transform(image)

        # 마스크 생성
        mask_np = self.mask_creator.random_mask(image_height=512, image_width=512)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()

        return {'image': image_tensor, 'mask': mask_tensor, 'path': image_path}

    def __len__(self):
        return self.dataset_size
