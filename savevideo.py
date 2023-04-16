import cv2
import os

output_name = "predictoutput_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

image_folder = "/home/worakan/save_images"
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
