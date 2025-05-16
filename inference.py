import sys
sys.argv += ['--model', 'arrange', '--netG', 'baseconv', '--dataset_mode', 'kadid']

import torch
from PIL import Image
from torchvision import transforms
from models.arrange_model import ArrangeModel
from options.test_options import TestOptions
import util.util as util
import os

# 1. 옵션 설정
opt = TestOptions().parse()
opt.isTrain = False
opt.no_fine_loss = False
opt.no_gan_loss = True
opt.no_vgg_loss = True
opt.load_pretrained_g = 'C:/Users/IIPL02/Desktop/inpainted_1/checkpoints/label2coco/10_net_G.pth'
opt.gpu_ids = [0] if torch.cuda.is_available() else []



# 2. 모델 초기화
model = ArrangeModel(opt)
model.eval()

# 3. weight 로드 (inpaint_model.py의 load_network_path 사용 가능)
model.netG = util.load_network_path(model.netG, opt.load_pretrained_g)

# 4. 전처리 정의
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform_mask = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 5. 이미지와 마스크 로딩
img_path = 'C:/Users/IIPL02/Desktop/inpainted_1/I26_23_04.png'
mask_path = 'C:/Users/IIPL02/Desktop/inpainted_1/input/test_mask.png'

image = transform_img(Image.open(img_path).convert('RGB')).unsqueeze(0)
mask = transform_mask(Image.open(mask_path).convert('L')).unsqueeze(0)
mask = (mask > 0.5).float()

if model.use_gpu():
    image = image.cuda()
    mask = mask.cuda()

# 6. 모델 입력 준비
data = {'image': image, 'mask': mask}

# 7. 모델 추론
with torch.no_grad():
    output, _ = model(data, mode='inference')
output = (output.clamp(-1, 1) + 1) / 2.0  # [-1, 1] → [0, 1]

# 8. 결과 저장
os.makedirs('output', exist_ok=True)
output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
output_image.save('output/inpainted_result.png')

print("✅ Inpainting 결과가 'output/inpainted_result.png'에 저장되었습니다.")

# python inference.py --model arrange --netG baseconv --dataset_mode kadid --which_epoch 1
