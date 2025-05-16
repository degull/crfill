import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt

        if gan_mode not in ['ls', 'original', 'w', 'hinge', 'softplus']:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.zero_tensor = torch.tensor(0.0, dtype=torch.float32, device=device)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.binary_cross_entropy_with_logits(input, target_tensor)
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    return -torch.mean(torch.min(input - 1, self.get_zero_tensor(input)))
                else:
                    return -torch.mean(torch.min(-input - 1, self.get_zero_tensor(input)))
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                return -torch.mean(input)
        elif self.gan_mode == 'softplus':
            return F.softplus(-input).mean() if target_is_real else F.softplus(input).mean()
        else:  # WGAN
            return -input.mean() if target_is_real else input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# ✅ 추가된 VGG 기반 Perceptual Loss
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features[:16]  # conv1_1 ~ conv4_1
        self.vgg = vgg.cuda(gpu_ids[0]) if gpu_ids else vgg
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        return self.criterion(x_vgg, y_vgg)
