#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from testmodel import DeblurDataset, try_dataloader
from skimage.metrics import structural_similarity
from math import log10, sqrt
import imutils
from imports.cane import cane_2d
import ctypes
import os
from tkinter import *
from tkinter import filedialog, ttk
import cv2
import numpy as np
import scipy.signal
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from PIL import Image, ImageTk, ImageChops, ImageEnhance
from skimage import filters
from skimage import color
from skimage.io import imsave
import skimage
import imageio
import torch
from torchvision.transforms import functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import time
from eval import _eval
from models.MIMOUNet import build_net
from torch.utils.data import Dataset, DataLoader
from PIL import Image as Image
from data import test_dataloader

ctypes.windll.shcore.SetProcessDpiAwareness(True)

root = Tk()
ttk.Style().configure("TButton", justify=CENTER)


gui_width = 1385
gui_height = 595
ip_file = ""
op_file = ""
image_dir=""
result_dir=""
original_img = None
modified_img = None
user_arg = None
popup = None
popup_input = None
cntr=0
ssimlisttest = []
ssimlistmodel = []
psnrlisttest = []
psnrlistmodel = []
SSIMbef=0
SSIMaft=0
PSNRbef=0
PSNRaft=0
test_flag=0



root.title("Image processing")
root.minsize(gui_width, gui_height)

#Podstawowe funkcje GUI

def set_user_arg():
    global user_arg
    user_arg = popup_input.get()
    popup.destroy()
    popup.quit()


def open_popup_input(text):
    global popup, popup_input
    popup = Toplevel(root)
    popup.resizable(False, False)
    popup.title("User Input")
    text_label = ttk.Label(popup, text=text, justify=LEFT)
    text_label.pack(side=TOP, anchor=W, padx=15, pady=10)
    popup_input = ttk.Entry(popup)
    popup_input.pack(side=TOP, anchor=NW, fill=X, padx=15)
    popup_btn = ttk.Button(popup, text="OK", command=set_user_arg).pack(pady=10)
    popup.geometry(f"400x{104+text_label.winfo_reqheight()}")
    popup_input.focus()
    popup.mainloop()


def draw_before_canvas():
    global original_img, ip_file
    original_img = Image.open(ip_file)
    original_img = original_img.convert("RGB")
    img = ImageTk.PhotoImage(original_img)
    before_canvas.create_image(
        256,
        256,
        image=img,
        anchor="center",
    )
    before_canvas.img = img

def draw_before_canv(img):
    original_img = img.convert("RGB")
    img = ImageTk.PhotoImage(original_img)
    before_canvas.create_image(
        256,
        256,
        image=img,
        anchor="center",
    )
    before_canvas.img = img


def draw_after_canvas(mimg):
    global modified_img

    modified_img = Image.fromarray(mimg)
    img = ImageTk.PhotoImage(modified_img)
    after_canvas.create_image(
        256,
        256,
        image=img,
        anchor="center",
    )
    after_canvas.img = img

def draw_after_canv(image):
    global modified_img

    modified_img = image
    img = ImageTk.PhotoImage(modified_img)
    after_canvas.create_image(
        256,
        256,
        image=img,
        anchor="center",
    )
    after_canvas.img = img   

def next_pic():
    global SSIMbef, SSIMaft, PSNRbef, PSNRaft, cntr 
    cntr+=1
    filelist = os.listdir(image_dir)
    filelist1 = os.listdir(result_dir)
    if cntr > len(filelist)-1 or  cntr < 0:
        cntr = 0
    path = os.path.join(image_dir, filelist[cntr])
    path1 = os.path.join(result_dir, filelist[cntr])
    image=Image.open(path)
    image1=Image.open(path1)
    if test_flag==1:
        SSIMa.delete(1.0,END)
        SSIMa.insert(END, 'SSIM:{}'.format(ssimlistmodel[cntr]))
        SSIMb.delete(1.0,END)
        SSIMb.insert(END,'SSIM:{}'.format(ssimlisttest[cntr]))
        PSNRa.delete(1.0,END)
        PSNRa.insert(END,'PSNR:{}'.format(psnrlistmodel[cntr]))
        PSNRb.delete(1.0,END)
        PSNRb.insert(END,'PSNR:{}'.format(psnrlisttest[cntr]))
    draw_before_canv(image)
    draw_after_canv(image1)
    

def prev_pic():
    global SSIMbef, SSIMaft, PSNRbef, PSNRaft, cntr 
    cntr-=1
    filelist = os.listdir(image_dir)
    filelist1 = os.listdir(result_dir)
    if cntr > len(filelist)-1 or  cntr < 0:
        cntr = 0
    path = os.path.join(image_dir, filelist[cntr])
    path1 = os.path.join(result_dir, filelist[cntr])
    image=Image.open(path)
    image1=Image.open(path1)
    if test_flag==1:
        SSIMa.delete(1.0,END)
        SSIMa.insert(END, 'SSIM:{}'.format(ssimlistmodel[cntr]))
        SSIMb.delete(1.0,END)
        SSIMb.insert(END,'SSIM:{}'.format(ssimlisttest[cntr]))
        PSNRa.delete(1.0,END)
        PSNRa.insert(END,'PSNR:{}'.format(psnrlistmodel[cntr]))
        PSNRb.delete(1.0,END)
        PSNRb.insert(END,'PSNR:{}'.format(psnrlisttest[cntr]))
    draw_before_canv(image)
    draw_after_canv(image1) 

    

def load_file():
    global ip_file
    ip_file = filedialog.askopenfilename(
        title="Open an image file",
        initialdir=".",
        filetypes=[("All Image Files", "*.*")],
    )
    draw_before_canvas()

def load_imagedir():
    global image_dir
    image_dir = filedialog.askdirectory(
    title="Choose image directory to open ",
    )
    

def load_resultdir():
    global result_dir
    result_dir = filedialog.askdirectory(
    title="Choose image directory to save ",
    )


def save_file():
    global ip_file, original_img, modified_img
    file_ext = os.path.splitext(ip_file)[1][1:]
    op_file = filedialog.asksaveasfilename(
        filetypes=[
            (
                f"{file_ext.upper()}",
                f"*.{file_ext}",
            )
        ],
        defaultextension=[
            (
                f"{file_ext.upper()}",
                f"*.{file_ext}",
            )
        ],
    )
    modified_img = modified_img.convert("RGB")
    modified_img.save(op_file)


left_frame = ttk.LabelFrame(root, text="Original Image", labelanchor=N)
left_frame.pack(fill=BOTH, side=LEFT, padx=10, pady=10, expand=1)

middle_frame = ttk.LabelFrame(root, text="Algorithms & Model", labelanchor=N)
middle_frame.pack(fill=BOTH, side=LEFT, padx=5, pady=10)

right_frame = ttk.LabelFrame(root, text="Modified Image", labelanchor=N)
right_frame.pack(fill=BOTH, side=LEFT, padx=10, pady=10, expand=1)

# left frame contents
before_canvas = Canvas(left_frame, bg="white", width=512, height=512)
before_canvas.pack(expand=1)

browse_btn = ttk.Button(left_frame, text="Browse", command=load_file)
browse_btn.pack(anchor=SW)

SSIMb = Text(left_frame, height = 1,
             width = 10)
SSIMb.pack(anchor=SE)

PSNRb = Text(left_frame, height = 1,
             width = 10)
PSNRb.pack(anchor=SE)



algo_canvas = Canvas(middle_frame, width=260, highlightthickness=0)
scrollable_algo_frame = Frame(algo_canvas)
scrollbar = Scrollbar(
    middle_frame, orient="vertical", command=algo_canvas.yview, width=15
)
scrollbar.pack(side="right", fill="y")
algo_canvas.pack(fill=BOTH, expand=1)
algo_canvas.configure(yscrollcommand=scrollbar.set)
algo_canvas.create_window((0, 0), window=scrollable_algo_frame, anchor="nw")
scrollable_algo_frame.bind(
    "<Configure>", lambda _: algo_canvas.configure(scrollregion=algo_canvas.bbox("all"))
)


after_canvas = Canvas(right_frame, bg="white", width=512, height=512)
after_canvas.pack(expand=1)

save_btn = ttk.Button(right_frame, text="Save", command=save_file)
save_btn.pack(expand=1, anchor=SE, pady=(5, 0))


SSIMa = Text(right_frame, height = 1,
              width = 10)
SSIMa.pack(anchor=SW)

PSNRa = Text(right_frame, height = 1,
              width = 10)
            
PSNRa.pack(anchor=SW)

#Obliczanie miar jakości

def SSIM(image_path, image_path2):
    imageA = cv2.imread(image_path)
    imageB = cv2.imread(image_path2)
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    (score, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    return score

def PSNR(orig_path, res_path):
    original = cv2.imread(orig_path)
    compressed = cv2.imread(res_path)
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        psnr=100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def testSet():
    global ssimlisttest
    global ssimlistmodel
    global psnrlisttest
    global psnrlistmodel

    if len(ssimlisttest)>0:
        ssimlisttest.clear()
    if len(ssimlistmodel)>0:
        ssimlistmodel.clear()
    if len(psnrlisttest)>0:
        ssimlisttest.clear()
    if len(psnrlistmodel)>0:
        ssimlistmodel.clear()
        
    global image_dir
    load_resultdir()
    image_dir = os.path.join(sys.path[0], 'blur')
    mask_dir = os.path.join(sys.path[0], 'masks')
    use_model_test(result_dir, image_dir)
    for filename in os.listdir(mask_dir):
        res=os.path.join(result_dir, filename)
        mask=os.path.join(mask_dir, filename)
        image=os.path.join(image_dir, filename)
        ssimlisttest.append(SSIM(image,mask))
        ssimlistmodel.append(SSIM(res,mask))
        psnrlisttest.append(PSNR(mask, image))
        psnrlistmodel.append(PSNR(mask, res))
            
    
#metody Preprocessingu

def sato():
    image = imageio.imread(ip_file)
    open_popup_input("Enter sigma value \n(Sigma should be an integer )")
    arg_list = user_arg.replace(" ", "").split(",")
    sigmas = int(arg_list[0])
    out = skimage.filters.sato(image, sigmas, black_ridges=True, mode="reflect")
    im = (out * 255).astype(np.uint8)
    im = skimage.exposure.rescale_intensity(im)
    draw_after_canvas(im) 
    return im

def callsato(image, sigmas):
    out = skimage.filters.sato(image, sigmas, black_ridges=True, mode="reflect")
    im = (out * 255).astype(np.uint8)
    im = skimage.exposure.rescale_intensity(im)
    draw_after_canvas(im) 
    return im

def contrast():
    image=Image.open(ip_file)
    open_popup_input("Enter contrast value \n(Bigger than 1.0 to enhance contrast )")
    arg_list = user_arg.replace(" ", "").split(",")
    factor = float(arg_list[0])
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(factor)
    draw_after_canv(image)
    return image


def callcontrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(factor)
    draw_after_canv(image)
    return image
    
def caefi2d():
    image = imageio.imread(ip_file)
    i = np.array(image)
    j = i.shape
    if len(j)>2:
        image = color.rgb2gray(image)
        
    image = image / np.max(image)
    open_popup_input("Enter smoothness \n(Smoothness should be double )")
    arg_list = user_arg.replace(" ", "").split(",")
    smooth_degree = float(arg_list[0])
    enhanced_image, enhance_hist = cane_2d(image, smooth_degree)
    im = (enhanced_image*255).astype(np.uint8)
    draw_after_canvas(im)
    return im

   
def callcaefi2d(image, smooth_degree):
    i = np.array(image)
    j = i.shape
    if len(j)>2:
        image = color.rgb2gray(image)
        
    image = image / np.max(image)
    enhanced_image, enhance_hist = cane_2d(image, smooth_degree)
    im = (enhanced_image*255).astype(np.uint8)
    draw_after_canvas(im)
    return im
    
def image_multiply():
    im2 = caefi2d()
    im1 = sato()
    im2 = Image.fromarray(im2)
    im1 = Image.fromarray(im1)
    im2 = im2.convert('RGB')
    im1 = ImageChops.invert(im1) 
    im = ImageChops.multiply(im1, im2)
    draw_after_canv(im)
    
def callimage_multiply(im1, im2):
    im = ImageChops.multiply(im1, im2)
    draw_after_canv(im)
    return im


def callRGB2Gray():
    grayscale = RGB2Gray()
    draw_after_canvas(grayscale)

def preprocessing():
    open_popup_input("Enter contrast, smoothness and sigma value: \n (Separated with a coma)")
    smoothness2= 0.0000035
    arg_list = user_arg.replace(" ", "").split(",")
    contrast = float(arg_list[0])
    smoothness = float(arg_list[1])
    sigmas = float(arg_list[2])
    image=Image.open(ip_file)
    im = callcontrast(image, contrast)
    im1 = callcaefi2d(im, smoothness)
    im2 = callsato(im1, sigmas)
    im1 = Image.fromarray(im1)
    im2 = Image.fromarray(im2)
    im2 = ImageChops.invert(im2) 
    im3 = callimage_multiply(im1,im2)
    img = callcaefi2d(im3, smoothness2)
          
def prepdefaultvalues():
    contrast = 1.5
    sigmas=2
    smoothness1=0.0000035
    smoothness2=0.0000035
    image=Image.open(ip_file)
    im = callcontrast(image, contrast)
    im1 = callcaefi2d(im, smoothness1)
    im2 = callsato(im1, sigmas)
    im1 = Image.fromarray(im1)
    im2 = Image.fromarray(im2)
    im2 = ImageChops.invert(im2) 
    im3 = callimage_multiply(im1,im2)
    img = callcaefi2d(im3, smoothness2)

#Użycie wytrenowanego modelu
          
def use_model():
    global cntr, test_flag
    test_flag=0
    cntr = 0
    load_imagedir()
    load_resultdir()
    model = build_net('MIMO-UNet')
    state_dict = torch.load('Best.pkl')
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    dataloader = try_dataloader(image_dir, batch_size=1, num_workers=0)

    with torch.no_grad():
        for iter_idx, data in enumerate(dataloader):
                    input_img, name = data
                    input_img = input_img.to(device)
                    pred = model(input_img)[2]
                    pred_clip = torch.clamp(pred, 0, 1)
                    save_name = os.path.join(result_dir, name[0])
                    pred_clip += 0.5 / 255
                    pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                    draw_after_canv(pred)
                    pred.save(save_name)
    imgp = os.path.join(image_dir, name[0])
    image = Image.open(imgp)
    draw_before_canv(image)
    
def use_model_test(result_dir, image_dir):
    global cntr, test_flag
    cntr = 0
    test_flag=1
    model = build_net('MIMO-UNet')
    state_dict = torch.load('Best.pkl')
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    dataloader = try_dataloader(image_dir, batch_size=1, num_workers=0)

    with torch.no_grad():
        for iter_idx, data in enumerate(dataloader):
                    input_img, name = data
                    input_img = input_img.to(device)
                    pred = model(input_img)[2]
                    pred_clip = torch.clamp(pred, 0, 1)
                    save_name = os.path.join(result_dir, name[0])
                    pred_clip += 0.5 / 255
                    pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                    draw_after_canv(pred)
                    pred.save(save_name)
    imgp = os.path.join(image_dir, name[0])
    image = Image.open(imgp)
    draw_before_canv(image)
    
    
    
                    
# Przyciski 


ttk.Button(
    scrollable_algo_frame,
    text="Contrast",
    width=30,
    command=contrast,
).pack(pady=2, ipady=2)


ttk.Button(
    scrollable_algo_frame,
    text="Sato Filter",
    width=30,
    command=sato,
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="CAEFI",
    width=30,
    command=caefi2d,
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Multiply Sato and CAEFI",
    width=30,
    command=image_multiply,
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Preprocessing",
    width=30,
    command=preprocessing,
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Preprocessing with default values",
    width=30,
    command=prepdefaultvalues,
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Use MIMO-UNet model",
    width=30,
    command=use_model,
).pack(pady=2, ipady=2)

ttk.Button(
    scrollable_algo_frame,
    text="Use model on a test set",
    width=30,
    command=testSet,
).pack(pady=2, ipady=2)

bn=ttk.Button(
    scrollable_algo_frame,
    text=">",
    width=5,
    command=next_pic,
).pack(anchor=CENTER)

bp=ttk.Button(
    scrollable_algo_frame,
    text="<",
    width=5,
    command=prev_pic,
).pack(anchor=CENTER)





root.mainloop()

