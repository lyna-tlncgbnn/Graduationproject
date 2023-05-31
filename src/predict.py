from paddle.vision.transforms import Resize
from run import opt
import numpy as np
import paddle

# 调用模型进行图像补全
def predict(img, mask,g):
    # 确保图像和mask的尺寸一致
    img = Resize([opt.img_size, opt.img_size], interpolation='bilinear')(img)
    mask = Resize([opt.img_size, opt.img_size], interpolation='nearest')(mask)
    img = img.convert('RGB')
    mask = mask.convert('L')
    img = np.array(img)
    mask = np.array(mask)

    img = (img.astype('float32') / 255.) * 2. - 1.
    img = np.transpose(img, (2, 0, 1))
    mask = np.expand_dims(mask.astype('float32') / 255., 0)
    img = paddle.to_tensor(np.expand_dims(img, 0))
    mask = paddle.to_tensor(np.expand_dims(mask, 0))

    # 预测
    img_masked = (img * (1 - mask)) + mask
    pred_img = g(img_masked, mask)
    comp_img = (1 - mask) * img + mask * pred_img

    # 将张量tensor格式转换为numpy图像格式，并将其值的范围从[-1,1]缩放到[0,1]
    # st.image()仅接收numpy类型的数据
    img_show = (comp_img.numpy()[0].transpose((1,2,0)) + 1.) / 2.
    img_masked_np = (img_masked.numpy()[0].transpose((1, 2, 0)) + 1.) / 2.
    
    return img_show,img_masked_np