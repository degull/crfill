import pdb
import torch
import models.networks as networks
import util.util as util
from models.create_mask import MaskCreator
import random
import numpy as np


class InpaintModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        parser.add_argument('--path_objectshape_base', type=str, default='', help='path obj base')
        parser.add_argument('--path_objectshape_list', type=str, default='', help='path obj list')
        parser.add_argument('--update_part', type=str, default='all', help='update part')
        parser.add_argument('--d_mask_in', action='store_true', help='if specified, d mask in')
        parser.add_argument('--no_fine_loss', action='store_true', help='if specified, do *not* use refinementstageloss')
        parser.add_argument('--load_pretrained_g', type=str, required=False, help='load pt g')
        parser.add_argument('--load_pretrained_d', type=str, required=False, help='load pt d')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() else torch.ByteTensor

        self.netG, self.netD = self.initialize_networks(opt)
        if opt.isTrain and opt.load_pretrained_g is not None:
            print(f"looad {opt.load_pretrained_g}")
            self.netG = util.load_network_path(self.netG, opt.load_pretrained_g)
        if opt.isTrain and opt.load_pretrained_d is not None:
            print(f"looad {opt.load_pretrained_d}")
            self.netD = util.load_network_path(self.netD, opt.load_pretrained_d)

        if opt.isTrain:
            self.mask_creator = MaskCreator()
            self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)

    def forward(self, data, mode):
        inputs, real_image, mask = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, coarse_image, composed_image = self.compute_generator_loss(inputs, real_image, mask)
            generated = {'coarse': coarse_image, 'composed': composed_image}
            return g_loss, inputs, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(inputs, real_image, mask)
            return d_loss, data['inputs']
        elif mode == 'inference':
            with torch.no_grad():
                coarse_image, fake_image = self.generate_fake(inputs, real_image, mask)
                composed_image = fake_image * mask + inputs * (1 - mask)
            return composed_image, inputs
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = self.netG.get_param_list(opt.update_part)
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        G_lr, D_lr = (opt.lr, opt.lr) if opt.no_TTUR else (opt.lr / 2, opt.lr * 2)

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
        return netG, netD

    def preprocess_input(self, data):
        b, c, h, w = data['image'].shape
        device = self.opt.gpu_ids[0] if self.use_gpu() else 'cpu'

        if self.opt.isTrain:
            mask1 = self.mask_creator.stroke_mask(h, w, max_length=min(h, w)/2)
            ri = random.randint(0, 3)
            if ri == 1 or ri == 0:
                mask2 = self.mask_creator.object_mask(h, w)
            else:
                mask2 = self.mask_creator.rectangle_mask(h, w, min(h, w) // 4, min(h, w) // 2)
            mask = (mask1 + mask2 > 0).astype(np.float32)
            mask = torch.tensor(mask, dtype=torch.float32, device=device)
            mask = mask[None, None, ...].expand(b, -1, -1, -1)
            data['mask'] = mask
        else:
            data['mask'] = data['mask'].to(device)
            mask = data['mask']

        data['image'] = data['image'].to(device)
        inputs = data['image'] * (1 - mask)
        data['inputs'] = inputs
        return inputs, data['image'], mask


    def g_image_loss(self, coarse_image, fake_image, composed_image, real_image, mask):
        G_losses = {}
        if not self.opt.no_gan_loss and not self.opt.no_fine_loss:
            pred_fake, pred_real = self.discriminate(composed_image, real_image, mask)
            G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)

        if not self.opt.no_vgg_loss and not self.opt.no_fine_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg

        if not self.opt.no_l1_loss:
            if coarse_image is not None:
                G_losses['L1c'] = torch.nn.functional.l1_loss(coarse_image, real_image) * self.opt.lambda_l1
            if not self.opt.no_fine_loss:
                G_losses['L1f'] = torch.nn.functional.l1_loss(fake_image, real_image) * self.opt.lambda_l1

        return G_losses

    def compute_generator_loss(self, inputs, real_image, mask):
        coarse_image, fake_image = self.generate_fake(inputs, real_image, mask)
        composed_image = fake_image * mask + inputs * (1 - mask)
        G_losses = self.g_image_loss(coarse_image, fake_image, composed_image, real_image, mask)
        return G_losses, coarse_image, composed_image

    def compute_discriminator_loss(self, inputs, real_image, mask):
        D_losses = {}
        if not self.opt.no_gan_loss:
            with torch.no_grad():
                coarse_image, fake_image = self.generate_fake(inputs, real_image, mask)
                fake_image = fake_image.detach()
                fake_image.requires_grad_()
                composed_image = fake_image * mask + inputs * (1 - mask)
            pred_fake, pred_real = self.discriminate(composed_image, real_image, mask)
            D_losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
            D_losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)
        return D_losses

    def generate_fake(self, inputs, real_image, mask):
        coarse_image, fake_image = self.netG(inputs, mask)
        return coarse_image, fake_image

    def discriminate(self, fake_image, real_image, mask):
        fake_and_real = torch.cat([fake_image, real_image], dim=0)
        mask_all = torch.cat([mask, mask], dim=0) if self.opt.d_mask_in else None
        discriminator_out = self.netD(fake_and_real, mask_all)
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        return pred_fake, pred_real

    def divide_pred(self, pred):
        if type(pred) == list:
            fake = [ [tensor[:tensor.size(0)//2] for tensor in p] for p in pred ]
            real = [ [tensor[tensor.size(0)//2:] for tensor in p] for p in pred ]
        else:
            fake = pred[:pred.size(0)//2]
            real = pred[pred.size(0)//2:]
        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] |= (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] |= (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] |= (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] |= (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
