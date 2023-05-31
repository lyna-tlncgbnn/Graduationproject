import os
import time
import math
import cv2
import numpy as np
from PIL import Image
from paddle.io import Dataset, DataLoader
from paddle.vision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, Resize
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.utils import spectral_norm
import paddle.distributed as dist
from paddle.io import DistributedBatchSampler
from visualdl import LogWriter


class OPT():
    def __init__(self):
        super(OPT, self).__init__()
        self.batch_size = 6  # V100

        self.img_size = 512
        self.rates = [1, 2, 4, 8]
        self.block_num = 8

        self.l1_weight = 1
        self.style_weight = 250
        self.perceptual_weight = .1
        self.adversal_weight = .01

        self.lr_g = 1e-4
        self.lr_d = 1e-4
        self.beta1 = .5
        self.beta2 = .999

        self.dataset_path = 'data/aot'
        self.output_path = 'output'
        self.vgg_weight_path = 'data/data210700/vgg19.pdparams'


opt = OPT()


# 定义数据集对象
class PlaceDataset(Dataset):
    def __init__(self, opt, istrain=True):
        super(PlaceDataset, self).__init__()

        self.image_path = []

        def get_all_sub_dirs(root_dir):
            file_list = []

            # 定义函数
            def get_sub_dirs(r_dir):
                for root, dirs, files in os.walk(r_dir):
                    if len(files) > 0:
                        for f in files:
                            file_list.append(os.path.join(root, f))
                    if len(dirs) > 0:
                        for d in dirs:
                            get_sub_dirs(os.path.join(root, d))
                    break

            # 调用函数
            get_sub_dirs(root_dir)
            return file_list

        # 设置训练集、验证集的数据存放路径
        if istrain:
            self.img_list = get_all_sub_dirs(
                os.path.join(opt.dataset_path, 'CelebA-HQ_512'))
            self.mask_dir = os.path.join(opt.dataset_path, 'train_mask')
        else:
            self.img_list = get_all_sub_dirs(
                os.path.join(opt.dataset_path, 'val/val_img'))
            self.mask_dir = os.path.join(opt.dataset_path, 'val/val_mask')

        self.img_list = np.sort(np.array(self.img_list))
        _, _, mask_list = next(os.walk(self.mask_dir))
        self.mask_list = np.sort(mask_list)

        self.istrain = istrain
        self.opt = opt

        # 数据增强
        if istrain:
            self.img_trans = Compose([
                # 将原图片随机裁剪出一块,再缩放成相应 (size*size) 的比例
                RandomResizedCrop(opt.img_size),
                RandomHorizontalFlip(),
                ColorJitter(0.05, 0.05, 0.05, 0.05),
            ])
            self.mask_trans = Compose([
                Resize([opt.img_size, opt.img_size], interpolation='nearest'),
                RandomHorizontalFlip(),
            ])
        else:
            self.img_trans = Compose([
                Resize([opt.img_size, opt.img_size], interpolation='bilinear'),
            ])
            self.mask_trans = Compose([
                Resize([opt.img_size, opt.img_size], interpolation='nearest'),
            ])

        self.istrain = istrain

    # 将送入模型的RGB图片，归一化到[-1,1]，形状为[n_b,c,h,w]
    # mask尺寸用于img一致，0对应已知像素，1表示缺失像素

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        mask = Image.open(
            os.path.join(
                self.mask_dir,
                self.mask_list[np.random.randint(0, self.mask_list.shape[0])]))
        img = self.img_trans(img)
        mask = self.mask_trans(mask)

        mask = mask.rotate(np.random.randint(0, 45))
        img = img.convert('RGB')
        mask = mask.convert('L')

        img = np.array(img).astype('float32')
        img = (img / 255.) * 2. - 1.
        img = np.transpose(img, (2, 0, 1))
        mask = np.array(mask).astype('float32') / 255.
        mask = np.expand_dims(mask, 0)

        return img, mask, self.img_list[idx]

    def __len__(self):
        return len(self.img_list)


# Aggragated Contextual Transformation Block：用于提取多尺度特征
class AOTBlock(nn.Layer):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()

        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)),
                nn.Sequential(
                    nn.Pad2D(rate, mode='reflect'),
                    nn.Conv2D(dim, dim // 4, 3, 1, 0, dilation=int(rate)),
                    nn.ReLU()))
        self.fuse = nn.Sequential(nn.Pad2D(1, mode='reflect'),
                                  nn.Conv2D(dim, dim, 3, 1, 0, dilation=1))
        self.gate = nn.Sequential(nn.Pad2D(1, mode='reflect'),
                                  nn.Conv2D(dim, dim, 3, 1, 0, dilation=1))

    def forward(self, x):
        out = [
            self.__getattr__(f'block{str(i).zfill(2)}')(x)
            for i in range(len(self.rates))
        ]
        out = paddle.concat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = F.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat


class UpConv(nn.Layer):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2D(inc, outc, 3, 1, 1)

    def forward(self, x):
        return self.conv(
            F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True))


# 生成器
class InpaintGenerator(nn.Layer):
    def __init__(self, opt):
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(nn.Pad2D(3, mode='reflect'),
                                     nn.Conv2D(4, 64, 7, 1, 0), nn.ReLU(),
                                     nn.Conv2D(64, 128, 4, 2, 1), nn.ReLU(),
                                     nn.Conv2D(128, 256, 4, 2, 1), nn.ReLU())

        self.middle = nn.Sequential(
            *[AOTBlock(256, opt.rates) for _ in range(opt.block_num)])

        self.decoder = nn.Sequential(UpConv(256, 128), nn.ReLU(),
                                     UpConv(128, 64), nn.ReLU(),
                                     nn.Conv2D(64, 3, 3, 1, 1))

    def forward(self, x, mask):
        x = paddle.concat([x, mask], 1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = paddle.tanh(x)

        return x


# 判别器
class Discriminator(nn.Layer):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        inc = 3
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2D(inc, 64, 4, 2, 1, bias_attr=False)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2D(64, 128, 4, 2, 1, bias_attr=False)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2D(128, 256, 4, 2, 1, bias_attr=False)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2D(256, 512, 4, 1, 1, bias_attr=False)),
            nn.LeakyReLU(0.2), nn.Conv2D(512, 1, 4, 1, 1))

    def forward(self, x):
        feat = self.conv(x)
        return feat


# 用于计算Perceptual Loss和Style Loss的vgg19模型（使用ImageNet预训练权重）
class VGG19F(nn.Layer):
    def __init__(self):
        super(VGG19F, self).__init__()

        self.feature_0 = nn.Conv2D(3, 64, 3, 1, 1)
        self.relu_1 = nn.ReLU()
        self.feature_2 = nn.Conv2D(64, 64, 3, 1, 1)
        self.relu_3 = nn.ReLU()

        self.mp_4 = nn.MaxPool2D(2, 2, 0)
        self.feature_5 = nn.Conv2D(64, 128, 3, 1, 1)
        self.relu_6 = nn.ReLU()
        self.feature_7 = nn.Conv2D(128, 128, 3, 1, 1)
        self.relu_8 = nn.ReLU()

        self.mp_9 = nn.MaxPool2D(2, 2, 0)
        self.feature_10 = nn.Conv2D(128, 256, 3, 1, 1)
        self.relu_11 = nn.ReLU()
        self.feature_12 = nn.Conv2D(256, 256, 3, 1, 1)
        self.relu_13 = nn.ReLU()
        self.feature_14 = nn.Conv2D(256, 256, 3, 1, 1)
        self.relu_15 = nn.ReLU()
        self.feature_16 = nn.Conv2D(256, 256, 3, 1, 1)
        self.relu_17 = nn.ReLU()

        self.mp_18 = nn.MaxPool2D(2, 2, 0)
        self.feature_19 = nn.Conv2D(256, 512, 3, 1, 1)
        self.relu_20 = nn.ReLU()
        self.feature_21 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_22 = nn.ReLU()
        self.feature_23 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_24 = nn.ReLU()
        self.feature_25 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_26 = nn.ReLU()

        self.mp_27 = nn.MaxPool2D(2, 2, 0)
        self.feature_28 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_29 = nn.ReLU()
        self.feature_30 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_31 = nn.ReLU()
        self.feature_32 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_33 = nn.ReLU()
        self.feature_34 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_35 = nn.ReLU()

    def forward(self, x):
        x = self.stand(x)
        feats = []
        group = []
        x = self.feature_0(x)
        x = self.relu_1(x)
        group.append(x)
        x = self.feature_2(x)
        x = self.relu_3(x)
        group.append(x)
        feats.append(group)

        group = []
        x = self.mp_4(x)
        x = self.feature_5(x)
        x = self.relu_6(x)
        group.append(x)
        x = self.feature_7(x)
        x = self.relu_8(x)
        group.append(x)
        feats.append(group)

        group = []
        x = self.mp_9(x)
        x = self.feature_10(x)
        x = self.relu_11(x)
        group.append(x)
        x = self.feature_12(x)
        x = self.relu_13(x)
        group.append(x)
        x = self.feature_14(x)
        x = self.relu_15(x)
        group.append(x)
        x = self.feature_16(x)
        x = self.relu_17(x)
        group.append(x)
        feats.append(group)

        group = []
        x = self.mp_18(x)
        x = self.feature_19(x)
        x = self.relu_20(x)
        group.append(x)
        x = self.feature_21(x)
        x = self.relu_22(x)
        group.append(x)
        x = self.feature_23(x)
        x = self.relu_24(x)
        group.append(x)
        x = self.feature_25(x)
        x = self.relu_26(x)
        group.append(x)
        feats.append(group)

        group = []
        x = self.mp_27(x)
        x = self.feature_28(x)
        x = self.relu_29(x)
        group.append(x)
        x = self.feature_30(x)
        x = self.relu_31(x)
        group.append(x)
        x = self.feature_32(x)
        x = self.relu_33(x)
        group.append(x)
        x = self.feature_34(x)
        x = self.relu_35(x)
        group.append(x)
        feats.append(group)

        return feats

    def stand(self, x):
        mean = paddle.to_tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
        std = paddle.to_tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])
        y = (x + 1.) / 2.
        y = (y - mean) / std
        return y


class L1():
    def __init__(self, ):
        self.calc = nn.L1Loss()

    def __call__(self, x, y):
        return self.calc(x, y)


# 计算原图片和生成图片通过vgg19模型各个层输出的激活特征图的L1 Loss
class Perceptual():
    def __init__(self, vgg, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(Perceptual, self).__init__()
        self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        x = F.interpolate(x, (opt.img_size, opt.img_size),
                          mode='bilinear',
                          align_corners=True)
        y = F.interpolate(y, (opt.img_size, opt.img_size),
                          mode='bilinear',
                          align_corners=True)
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        content_loss = 0.0
        for i in range(len(self.weights)):
            content_loss += self.weights[i] * self.criterion(
                x_features[i][0],
                y_features[i][0])  # 此vgg19预训练模型无bn层，所以尝试不用rate
        return content_loss


# 通过vgg19模型，计算原图片与生成图片风格相似性的Loss
class Style():
    def __init__(self, vgg):
        super(Style, self).__init__()
        self.vgg = vgg
        self.criterion = nn.L1Loss()

    def compute_gram(self, x):
        b, c, h, w = x.shape
        f = x.reshape([b, c, w * h])
        f_T = f.transpose([0, 2, 1])
        G = paddle.matmul(f, f_T) / (h * w * c)
        return G

    def __call__(self, x, y):
        x = F.interpolate(x, (opt.img_size, opt.img_size),
                          mode='bilinear',
                          align_corners=True)
        y = F.interpolate(y, (opt.img_size, opt.img_size),
                          mode='bilinear',
                          align_corners=True)
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        style_loss = 0.0
        blocks = [2, 3, 4, 5]
        layers = [2, 4, 4, 2]
        for b, l in list(zip(blocks, layers)):
            b = b - 1
            l = l - 1
            style_loss += self.criterion(self.compute_gram(x_features[b][l]),
                                         self.compute_gram(y_features[b][l]))
        return style_loss


# 对叠加在图片上的mask边缘进行高斯模糊处理
def gaussian_blur(input, kernel_size, sigma):
    def get_gaussian_kernel(kernel_size: int, sigma: float) -> paddle.Tensor:
        def gauss_fcn(x, window_size, sigma):
            return -(x - window_size // 2)**2 / float(2 * sigma**2)

        gauss = paddle.stack([
            paddle.exp(paddle.to_tensor(gauss_fcn(x, kernel_size, sigma)))
            for x in range(kernel_size)
        ])
        return gauss / gauss.sum()

    b, c, h, w = input.shape
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d = paddle.matmul(kernel_x, kernel_y, transpose_y=True)
    kernel = kernel_2d.reshape([1, 1, ksize_x, ksize_y])
    kernel = kernel.repeat_interleave(c, 0)
    padding = [(k - 1) // 2 for k in kernel_size]
    return F.conv2d(input, kernel, padding=padding, stride=1, groups=c)


# GAN Loss，采用最小二乘Loss
class Adversal():
    def __init__(self, ksize=71):
        self.ksize = ksize
        self.loss_fn = nn.MSELoss()

    def __call__(self, netD, fake, real, masks):
        fake_detach = fake.detach()

        g_fake = netD(fake)
        d_fake = netD(fake_detach)
        d_real = netD(real)

        _, _, h, w = g_fake.shape
        b, c, ht, wt = masks.shape

        # 对齐判别器输出特征图与mask的尺寸
        if h != ht or w != wt:
            masks = F.interpolate(masks,
                                  size=(h, w),
                                  mode='bilinear',
                                  align_corners=True)
        d_fake_label = gaussian_blur(1 - masks, (self.ksize, self.ksize),
                                     (10, 10)).detach()
        d_real_label = paddle.ones_like(d_real)
        g_fake_label = paddle.ones_like(g_fake)

        dis_loss = [
            self.loss_fn(d_fake, d_fake_label).mean(),
            self.loss_fn(d_real, d_real_label).mean()
        ]
        gen_loss = (self.loss_fn(g_fake, g_fake_label) * masks /
                    paddle.mean(masks)).mean()

        return dis_loss, gen_loss


# 初始化输出文件夹（默认为项目路径下的output/文件夹
def init_output(output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        # 记录当前迭代步数
        current_step = np.array([0])
        np.save(os.path.join(output_path, "current_step"), current_step)
        print('训练输出目录[' + output_path + ']初始化完成')
    # 存储生成器、判别器check point
    if not os.path.exists(os.path.join(output_path, "model")):
        os.mkdir(os.path.join(output_path, "model"))
    # 存储训练时生成的图片
    if not os.path.exists(os.path.join(output_path, 'pic')):
        os.mkdir(os.path.join(output_path, 'pic'))
    # 存储预测时生成的图片
    if not os.path.exists(os.path.join(output_path, 'pic_val')):
        os.mkdir(os.path.join(output_path, 'pic_val'))


log_writer = LogWriter("./log/aot")

def train(show_interval=100,
          save_interval=10,
          total_iter=100000000,
          epoch_num=50):
    # 初始化训练输出路径
    init_output(opt.output_path)

    # 设置支持多卡训练
    dist.init_parallel_env()

    # 读取当前训练进度
    current_step = np.load(os.path.join(opt.output_path,
                                        'current_step.npy'))[0]
    print('已经完成 [' + str(current_step) + '] 步训练，开始继续训练...')

    # 定义数据读取用的DataLoader
    pds = PlaceDataset(opt)
    batchsamp = DistributedBatchSampler(pds,
                                        shuffle=True,
                                        batch_size=opt.batch_size,
                                        drop_last=True)
    loader = DataLoader(pds, batch_sampler=batchsamp, num_workers=4)
    data_total_num = pds.__len__()

    # 初始化生成器、判别器、计算Perceptu Loss用的VGG19模型（权重迁移自PyTorch）
    # vgg模型不参与训练，设为预测模式
    g = InpaintGenerator(opt)
    g = paddle.DataParallel(g)
    d = Discriminator()
    d = paddle.DataParallel(d)

    vgg19 = VGG19F()
    vgg_state_dict = paddle.load(opt.vgg_weight_path)
    vgg19.set_state_dict(vgg_state_dict)
    g.train()
    d.train()
    vgg19.eval()

    # 定义优化器
    opt_g = paddle.optimizer.Adam(learning_rate=opt.lr_g,
                                  beta1=opt.beta1,
                                  beta2=opt.beta2,
                                  parameters=g.parameters())
    opt_d = paddle.optimizer.Adam(learning_rate=opt.lr_d,
                                  beta1=opt.beta1,
                                  beta2=opt.beta2,
                                  parameters=d.parameters())

    # 如果当前步骤大于 0，加载保存的模型权重和优化器参数；否则，加载预训练权重
    if current_step > 0:
        print('读取存储的模型权重、优化器参数...')
        para = paddle.load(os.path.join(opt.output_path, "model/g.pdparams"))
        g.set_state_dict(para)
        para = paddle.load(os.path.join(opt.output_path, "model/d.pdparams"))
        d.set_state_dict(para)
        para = paddle.load(os.path.join(opt.output_path, "model/g.pdopt"))
        opt_g.set_state_dict(para)
        para = paddle.load(os.path.join(opt.output_path, "model/d.pdopt"))
        opt_d.set_state_dict(para)

    # 定义各部分loss
    l1_loss = L1()
    perceptual_loss = Perceptual(vgg19)
    style_loss = Style(vgg19)
    adv_loss = Adversal()

    # 设置训练时生成图片的存储路径
    pic_path = os.path.join(opt.output_path, 'pic')

    # 训练循环
    for epoch in range(epoch_num):
        start = time.time()
        if current_step >= total_iter:
            break
        for step, data in enumerate(loader):
            if current_step >= total_iter:
                break
            current_step += 1

            # 给图片加上mask
            img, mask, fname = data
            img_masked = (img * (1 - mask)) + mask
            pred_img = g(img_masked, mask)
            comp_img = (1 - mask) * img + mask * pred_img

            # 模型参数更新过程
            loss_g = {}
            loss_g['l1'] = l1_loss(img, pred_img) * opt.l1_weight
            loss_g['perceptual'] = perceptual_loss(
                img, pred_img) * opt.perceptual_weight
            loss_g['style'] = style_loss(img, pred_img) * opt.style_weight
            dis_loss, gen_loss = adv_loss(d, comp_img, img, mask)
            loss_g['adv_g'] = gen_loss * opt.adversal_weight
            loss_g_total = loss_g['l1'] + loss_g['perceptual'] + loss_g[
                'style'] + loss_g['adv_g']
            loss_d_fake = dis_loss[0]
            loss_d_real = dis_loss[1]
            loss_d_total = loss_d_fake + loss_d_real
            opt_g.clear_grad()
            opt_d.clear_grad()
            loss_g_total.backward()
            loss_d_total.backward()
            opt_g.step()
            opt_d.step()

            if current_step % save_interval == 0:
                log_writer.add_scalar(tag='train/g_total',
                                      step=current_step,
                                      value=loss_g_total.numpy())
                log_writer.add_scalar(tag='train/d_total',
                                      step=current_step,
                                      value=loss_d_total.numpy())
                log_writer.add_scalar(tag='train/g_l1',
                                      step=current_step,
                                      value=loss_g['l1'].numpy())
                log_writer.add_scalar(tag='train/g_perceptual',
                                      step=current_step,
                                      value=loss_g['perceptual'].numpy())
                log_writer.add_scalar(tag='train/g_style',
                                      step=current_step,
                                      value=loss_g['style'].numpy())
                log_writer.add_scalar(tag='train/d_fake',
                                      step=current_step,
                                      value=dis_loss[0].numpy())
                log_writer.add_scalar(tag='train/d_real',
                                      step=current_step,
                                      value=dis_loss[1].numpy())

            # 写log文件，保存生成的图片，定期保存模型check point
            log_interval = 1 if current_step < 10000 else 100
            if dist.get_rank() == 0:  # 只在主进程执行
                if current_step % log_interval == 0:
                    logfn = 'log.txt'
                    f = open(os.path.join(opt.output_path, logfn), 'a')
                    logtxt = 'current_step:[' + str(
                        current_step) + ']\t' + 'g_l1:' + str(
                            loss_g['l1'].numpy()
                        ) + '\t' + 'g_perceptual:' + str(
                            loss_g['perceptual'].numpy()
                        ) + '\t' + 'g_style:' + str(loss_g['style'].numpy(
                        )) + '\t' + 'g_adversal:' + str(loss_g['adv_g'].numpy(
                        )) + '\t' + 'g_total:' + str(
                            loss_g_total.numpy()) + '\t' + 'd_fake:' + str(
                                loss_d_fake.numpy()) + '\t' + 'd_real:' + str(
                                    loss_d_real.numpy()
                                ) + '\t' + 'd_total:' + str(loss_d_total.numpy(
                                )) + '\t' + 'filename:[' + fname[
                                    0] + ']\t' + 'time:[' + time.strftime(
                                        '%Y-%m-%d %H:%M:%S',
                                        time.localtime(time.time())) + ']\n'
                    f.write(logtxt)
                    f.close()

                # show img
                if current_step % show_interval == 0:
                    print(
                        'current_step:', current_step, 'epoch:', epoch,
                        'step:[' + str(step) + '/' +
                        str(math.ceil(data_total_num / opt.batch_size)) + ']'
                        'g_l1:', loss_g['l1'].numpy(), 'g_perceptual:',
                        loss_g['perceptual'].numpy(), 'g_style:',
                        loss_g['style'].numpy(), 'g_adversal:',
                        loss_g['adv_g'].numpy(), 'g_total:',
                        loss_g_total.numpy(), 'd_fake:', loss_d_fake.numpy(),
                        'd_real:', loss_d_real.numpy(), 'd_total:',
                        loss_d_total.numpy(), 'filename:', fname[0],
                        time.strftime('%Y-%m-%d %H:%M:%S',
                                      time.localtime(time.time())))

                    img_show2 = (pred_img.numpy()[0].transpose(
                        (1, 2, 0)) + 1.) / 2.
                    img_show2 = (img_show2 * 256).astype('uint8')
                    img_show2 = cv2.cvtColor(img_show2, cv2.COLOR_RGB2BGR)
                    img_show4 = (mask.numpy()[0][0] * 255).astype('uint8')
                    cv2.imwrite(
                        os.path.join(pic_path,
                                     os.path.split(fname[0])[1]), img_show2)
                    cv2.imwrite(
                        os.path.join(
                            pic_path,
                            os.path.split(fname[0])[1].replace('.', '_mask.')),
                        img_show4)

                # 定时存盘
                if current_step % save_interval == 0:
                    para = g.state_dict()
                    paddle.save(
                        para, os.path.join(opt.output_path,
                                           "model/g.pdparams"))
                    para = d.state_dict()
                    paddle.save(
                        para, os.path.join(opt.output_path,
                                           "model/d.pdparams"))
                    para = opt_g.state_dict()
                    paddle.save(para,
                                os.path.join(opt.output_path, "model/g.pdopt"))
                    para = opt_d.state_dict()
                    paddle.save(para,
                                os.path.join(opt.output_path, "model/d.pdopt"))
                    np.save(os.path.join(opt.output_path, 'current_step'),
                            np.array([current_step]))
                    print('第[' + str(current_step) + ']步模型保存。保存路径：',
                          os.path.join(opt.output_path, "model"))

            # 存储clock
            if current_step % 10 == 0:
                clock = np.array([
                    str(current_step),
                    time.strftime('%Y-%m-%d %H:%M:%S',
                                  time.localtime(time.time()))
                ])
                np.savetxt(os.path.join(opt.output_path, 'clock.txt'),
                           clock,
                           fmt='%s',
                           delimiter='\t')

    # 训练迭代完成时保存模型参数
    para = g.state_dict()
    paddle.save(para, os.path.join(opt.output_path, "model/g.pdparams"))
    para = d.state_dict()
    paddle.save(para, os.path.join(opt.output_path, "model/d.pdparams"))
    para = opt_g.state_dict()
    paddle.save(para, os.path.join(opt.output_path, "model/g.pdopt"))
    para = opt_d.state_dict()
    paddle.save(para, os.path.join(opt.output_path, "model/d.pdopt"))
    np.save(os.path.join(opt.output_path, 'current_step'),
            np.array([current_step]))
    print('第[' + str(current_step) + ']步模型保存。保存路径：',
          os.path.join(opt.output_path, "model"))
    print('Finished training! Total Iteration:', current_step)


if __name__ == '__main__':
    train()