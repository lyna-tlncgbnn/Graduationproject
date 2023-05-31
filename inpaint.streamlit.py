import streamlit as st
import os
import numpy as np
from PIL import Image
import paddle
import paddle.nn as nn
from paddle.vision.transforms import Resize
from run import InpaintGenerator,opt
# from run_coarse2fine import InpaintGenerator,opt
from src.metrics_single.metrics_l1_psnr_ssim import calculate_psnr,calculate_ssim,calculate_l1

import pandas as pd

def resize_image(image, size):
    """将图像调整为给定的大小"""
    resized_image = image.resize(size)
    return resized_image

# 设置宽屏模式
st.set_page_config(page_title="基于生成对抗网络和多尺度特征融合模块的高分辨率图像补全系统———2019211775-姚镜池", layout="wide")

st.title("基于生成对抗网络和多尺度特征融合模块的高分辨率图像补全系统———2019211775-姚镜池")

st.sidebar.info("文件上传区")
# 侧边栏排版，上传图像
img = st.sidebar.file_uploader("请上传待补全的图像", type=['png', 'jpg'])
if img is not None:
    image = Image.open(img)
    image_resized = resize_image(image, (512,512))
mask = st.sidebar.file_uploader("请上传对应的掩码图像", type=['png', 'jpg'])
if mask is not None:
    mask2 = Image.open(mask)
    mask_resized = resize_image(mask2, (512,512))

# 展示原始图像和Mask
col3, col4 = st.columns(2)
with col3:
    if img:
        st.image(image_resized,caption='原始图像/GT')
    # 默认图像
    else:
        img='val/val_img/252027220.jpg'
        image =  Image.open(img)
        image_resized = resize_image(image, (512,512))
        st.image(image_resized,caption='原始图像/GT')
with col4:
    if mask:
        st.image(mask_resized,caption='二值掩码图像/MASK')
    # 默认mask
    else:
        mask='val/val_mask/252027220.png'
        mask2 = Image.open(mask)
        mask_resized = resize_image(mask2, (512,512))
        st.image(mask_resized,caption='二值掩码图像/MASK')

# button
col5, col6, col7 = st.columns([1.2,1,1])
with col6:
    test=st.button("开始进行图像补全")

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

col8, col9 = st.columns(2)
if test:
    if img is not None and mask is not None:
        # 加载模型
        g = InpaintGenerator(opt)
        g.eval()
        model_path = "model/g.pdparams"
        # model_path = "output_coarse2fine/model/g.pdparams" 
        
        para = paddle.load(model_path)
        g.set_state_dict(para)

        # 设定图像尺寸
        opt.img_size = 512
        img = Image.open(img)
        mask = Image.open(mask)
        output,img_masked_np = predict(img, mask, g)
        
        # 数据对齐，将output从numpy类型转换为PIL类型
        output_scaled = (output * 255).astype('uint8')
        output_image = Image.fromarray(output_scaled)

        # st.sidebar.write("img1 type:",type(image_resized),image_resized.mode,'\n\n',"img2 type: ",type(output_image),output_image.mode)
        res_l1 = round(calculate_l1(image_resized, output_image),4)
        res_psnr = round(calculate_psnr(image_resized, output_image),4)
        res_ssim = round(calculate_ssim(image_resized, output_image),4)
        
        # st.sidebar.metric("PSNR", res_psnr)
        # st.sidebar.metric("SSIM",res_ssim)
        # st.sidebar.metric("MAE", res_l1)

        # 补全成功，界面展示
        st.balloons()
        with col8:
            st.image(img_masked_np,caption='破损图像')
        with col9: 
            st.image(output,caption='补全图像')

        # """柱状图"""
        # # 定义数值
        # data = {"PSNR": res_psnr, "SSIM": res_ssim, "MAE": res_l1}

        # # 将数据转换为 pandas DataFrame
        # df = pd.DataFrame(data.items(), columns=["Metric", "Value"])

        # # 使用 st.bar_chart() 函数进行可视化
        # st.bar_chart(df.set_index("Metric"))

        st.sidebar.info("客观评估指标")
        # 定义数值
        psnr = res_psnr
        ssim = res_ssim
        mae = res_l1

        # 使用自定义样式进行展示
        st.markdown("""
        <style>
        .custom-value {
            font-size: 24px;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

        st.sidebar.markdown(f'<div class="custom-value">PSNR: {psnr}</div>', unsafe_allow_html=True)
        st.sidebar.markdown(f'<div class="custom-value">SSIM: {ssim}</div>', unsafe_allow_html=True)
        st.sidebar.markdown(f'<div class="custom-value">MAE: {mae}</div>', unsafe_allow_html=True)


    else:
        st.warning("请确保已上传图片和掩码")
else:
    st.info("暂无预测结果,请点击提交图片")