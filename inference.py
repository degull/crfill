import torch
from PIL import Image
from torchvision import transforms
from models.arrange_model import ArrangeModel
from options.test_options import TestOptions
import util.util as util

# 옵션 정의
opt = TestOptions().parse()
opt.isTrain = False
opt.no_fine_loss = False
opt.no_gan_loss = True
opt.no_vgg_loss = True
opt.load_pretrained_g = 'checkpoints/your_model.pth'

# 모델 불러오기
model = ArrangeModel(opt)
model.eval()

# 이미지 전처리
img_path = 'input/test_img.jpg'
mask_path = 'input/test_mask.png'

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

image = transform(Image.open(img_path).convert('RGB')).unsqueeze(0)
mask = transform(Image.open(mask_path).convert('L')).unsqueeze(0)
mask = (mask > 0.5).float()

data = {'image': image, 'mask': mask}
output, _ = model(data, mode='inference')
output = (output.clamp(-1, 1) + 1) / 2.0  # to [0,1]

# 저장
output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
output_image.save('output/inpainted_result.png')
