import os
from os.path import join
# os.getcwd()
# "/home/worakan/miniconda3/envs/cansat"
# os.chdir("/home/worakan/miniconda3/envs/cansat/STEGO/src/")
# os.getcwd()
# cwd = "/home/worakan/miniconda3/envs/cansat/STEGO/src/"
# if cwd == os.getcwd():
 
# # print the current directory
#     print("Current working directory is:", cwd)
# else:
#     print(os.getcwd())
saved_models_dir = join("..", "saved_models")
os.makedirs(saved_models_dir, exist_ok=True)



import wget
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import cv2

saved_model_url_root = "https://marhamilresearch4.blob.core.windows.net/stego-public/saved_models/"
saved_model_name = "cocostuff27_vit_base_5.ckpt"
if not os.path.exists(join(saved_models_dir, saved_model_name)):
  wget.download(saved_model_url_root + saved_model_name, join(saved_models_dir, saved_model_name))

from train_segmentation import LitUnsupervisedSegmenter
from torchvision.transforms.functional import to_tensor
from utils import get_transform
from utils import unnorm, remove_axes
import torch.nn.functional as F
from crf import dense_crf
import torch
model = LitUnsupervisedSegmenter.load_from_checkpoint(join(saved_models_dir, saved_model_name)).cuda()



img_url ="/home/worakan/deepglobe-road-extraction-dataset/test/297_sat.jpg"
# response = requests.get(img_url)
img = Image.open(img_url)
transform = get_transform(448, False, "center")
img = transform(img).unsqueeze(0).cuda()

with torch.no_grad():
  code1 = model(img)
  code2 = model(img.flip(dims=[3]))
  code  = (code1 + code2.flip(dims=[3])) / 2
  code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)
  linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
  cluster_probs = model.cluster_probe(code, 3, log_probs=True).cpu()

  single_img = img[0].cpu()
  linear_pred = dense_crf(single_img, linear_probs[0]).argmax(0)
  cluster_pred = dense_crf(single_img, cluster_probs[0]).argmax(0)


n_classes = model.label_cmap.shape[0]
rgb_colors = model.label_cmap[linear_pred].reshape(-1, 3)

n_unique_colors = len(np.unique(rgb_colors, axis=0))
# print(f"Number of unique RGB colors in the predicted labels: {n_unique_colors}")
# print(np.unique(rgb_colors, axis=0))
import matplotlib.pyplot as plt
from utils import unnorm, remove_axes
# fig, ax = plt.subplots(1,3, figsize=(5*3,5))
# ax[0].imshow(unnorm(img)[0].permute(1,2,0).cpu())
# ax[0].set_title("Image")
# ax[1].imshow(model.label_cmap[cluster_pred])
# ax[1].set_title("Cluster Predictions")
# ax[2].imshow(model.label_cmap[linear_pred])
# ax[2].set_title("Linear Probe Predictions")
# remove_axes(ax)

# assume model.label_cmap is a colormap array of shape (n_classes, 3)
# and linear_pred is a 2D array of predicted label values of shape (height, width)
n_classes = model.label_cmap.shape[0]
rgb_colors = model.label_cmap[linear_pred].reshape(-1, 3)

# find the unique RGB colors in rgb_colors
unique_colors = np.unique(rgb_colors, axis=0)

# calculate the percentage of each unique RGB color
# percentages = []
# labels = ["Forest","Paddy field or open space","Forest  or Paddy field or open space","empty space in the forest","buildings"]
# colors = []
# for color in unique_colors:
#     n_color = np.sum(np.all(rgb_colors == color, axis=1))
#     percent_color = 100 * n_color / rgb_colors.shape[0]
#     percentages.append(percent_color)
#     colors.append(color/255)

# # sort the percentages in descending order and select the top 5
# sorted_percentages, sorted_colors = zip(*sorted(zip(percentages, colors), reverse=True))
# top5_percentages = sorted_percentages[:5]
# top5_colors = sorted_colors[:5]

# # plot the pie chart of RGB color distribution for the top 5 percentages
# plt.pie(top5_percentages, labels=labels, colors=top5_colors)
# plt.title('Land use Distribution (Top 5)')
# plt.axis('equal')
# plt.legend(top5_percentages)
# plt.show()

input_folder_path = "/home/worakan/save_unsupervised/"
output_folder_path = "/home/worakan/vsave_unsupervised/output/"


save_dir = '/home/worakan/vsave_unsupervised/chart'

# create output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
count = 0
# loop over all images in input folder
for filename in os.listdir(input_folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        try:

            # load image and apply transformation
            img = Image.open(os.path.join(input_folder_path, filename))
            transform = get_transform(448, False, "center")
            img = transform(img).unsqueeze(0).cuda()

            # apply model and CRF
            with torch.no_grad():
                code1 = model(img)
                code2 = model(img.flip(dims=[3]))
                code  = (code1 + code2.flip(dims=[3])) / 2
                code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)
                linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
                cluster_probs = model.cluster_probe(code, 3, log_probs=True).cpu()
                single_img = img[0].cpu()
                linear_pred = dense_crf(single_img, linear_probs[0]).argmax(0)
                cluster_pred = dense_crf(single_img, cluster_probs[0]).argmax(0)

            linear_pred_img = Image.fromarray(model.label_cmap[linear_pred].astype('uint8'))
            linear_pred_img.save(os.path.join(output_folder_path, filename))

            n_classes = model.label_cmap.shape[0]

            rgb_colors = model.label_cmap[linear_pred].reshape(-1, 3)
            unique_colors = np.unique(rgb_colors, axis=0)


            percentages = []
            labels = ["Forest","Paddy field or open space","Forest  or Paddy field or open space","empty space in the forest","buildings"]
            colors = []
            for color in unique_colors:
                n_color = np.sum(np.all(rgb_colors == color[np.newaxis, :], axis=1))
                percent_color = 100 * n_color / rgb_colors.shape[0]
                percentages.append(percent_color)
                colors.append(color/255)

            # sort the percentages in descending order and select the top 5
            sorted_percentages, sorted_colors = zip(*sorted(zip(percentages, colors), reverse=True))
            top5_percentages = sorted_percentages[:5]
            top5_colors = sorted_colors[:5]

            # plot the pie chart of RGB color distribution for the top 5 percentages
            plt.pie(top5_percentages, labels=labels, colors=top5_colors)
            plt.title('Land use Distribution (Top 5)')
            plt.axis('equal')
            plt.legend(top5_percentages)
            plt.savefig(os.path.join(save_dir, f'prediction_{count}.jpg'))
            # plt.show()
            count += 1
            print(f'fream{count}')

        except ValueError:
            pass

output_name = "output_video_unsupervies.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

image_folder = "/home/worakan/vsave_unsupervised/output"

image_names = sorted(os.listdir(image_folder))

image_path = os.path.join(image_folder, image_names[0])
image = cv2.imread(image_path)
height, width, channels = image.shape

video = cv2.VideoWriter(output_name, fourcc, 30, (width, height))

for image_name in image_names:
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    video.write(image)

video.release()
cv2.destroyAllWindows()

output_name = "output_unsupervisedvideo_chart.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

image_folder = "/home/worakan/vsave_unsupervised/chart"
image_names = sorted(os.listdir(image_folder))

image_path = os.path.join(image_folder, image_names[0])
image = cv2.imread(image_path)
height, width, channels = image.shape

video = cv2.VideoWriter(output_name, fourcc, 30, (width, height))

for image_name in image_names:
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    video.write(image)

video.release()
cv2.destroyAllWindows()
