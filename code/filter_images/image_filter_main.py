from filter_images.image_reader import ReadImage
from patch_matcher.patch_matcher import AdvancePatchMatcher
from utils.glob_def import DATA_DIR
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from scipy import signal, ndimage

if __name__ == "__main__":
    # load 1 pixel images noise
    image_reader = ReadImage(DATA_DIR, '8.txt', 10, [])

    template_image_path = os.path.join(DATA_DIR,"set","map.png")
    template = Image.open(template_image_path)
    patch_matcher = AdvancePatchMatcher(template)

    while image_reader.is_there_next_image():
        # read next image pair
        image, gt_image = image_reader.read_next_image_pair()

        patch_matcher.extract_key_points(image)
        patch_matcher.extract_key_points(gt_image)
        gaus_image = ndimage.gaussian_filter(image, sigma = 1, truncate = 4)
        patch_matcher.extract_key_points(gaus_image)

        plt.show()
        # show images
        plt.figure('image')
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)

        plt.figure('filt image')
        plt.imshow(gaus_image, cmap='gray', vmin=0, vmax=255)

        plt.figure('gt_image')
        plt.imshow(gt_image, cmap='gray', vmin=0, vmax=255)

        plt.figure('diff gt-img')
        plt.imshow(gt_image - image, cmap='gray', vmin=0, vmax=255)

        plt.show()