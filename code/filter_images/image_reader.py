import os
import numpy as np
from PIL import Image

np.random.seed(100)

class ReadImage:

    def __init__(self, path_to_dataset: str, txt_file_name: str, num_images: int, img_indeces: list):
        # setup paths
        self.dataset_path = path_to_dataset
        self.path_to_input_txt_files = os.path.join(path_to_dataset, 'inputs')
        self.path_to_onput_txt_files = os.path.join(path_to_dataset, 'outputs')
        self.path_to_input_patches = os.path.join(path_to_dataset, 'set')

        self.path_to_input_txt = os.path.join(self.path_to_input_txt_files, txt_file_name)
        self.path_to_gt_txt = os.path.join(self.path_to_onput_txt_files, txt_file_name)

        # load input file
        input_file = open(self.path_to_input_txt, 'r')
        self.input_content = input_file.readlines()
        self.input_offset = 3
        self.num_images = int(self.input_content[1])
        patch_size_str = self.input_content[2].split()
        self.ph = int(patch_size_str[0])
        self.pw = int(patch_size_str[1])

        # load ground truth file
        gt_file = open(self.path_to_gt_txt, 'r')
        self.gt_content = gt_file.readlines()

        # get random n images
        self.num_images =  num_images
        if len(img_indeces) == 0:
            num_random_images = num_images
            self.image_inds = np.round(np.random.rand(num_random_images) * self.num_images)
        else:
            self.image_inds = img_indeces

        # current image is first image
        self.curr_image_ind = 0

        # load teamplate image
        # proces template image
        template_image_path = os.path.join(self.dataset_path, "set", "map.png")
        template = Image.open(template_image_path)
        self.template = np.array(template.convert('L'))

    def is_there_next_image(self):
        return self.curr_image_ind < self.num_images

    def read_next_image_pair(self):

        img_index = self.curr_image_ind % self.num_images
        # get relative path to patch
        path_to_patch = self.input_content[self.input_offset + img_index]

        # take relative path from dataset path
        path_to_patch = path_to_patch.split('/')
        path_to_patch = os.path.join(path_to_patch[1], path_to_patch[2])

        # delete new line read from input
        path_to_patch = path_to_patch[:-1]

        # make full path to patch
        path_to_patch = os.path.join(self.path_to_input_patches, path_to_patch)

        # read patch
        patch = Image.open(path_to_patch)
        patch = np.array(patch.convert('L'))

        # read gt image
        # get relative path to patch
        gt_data_pos = self.gt_content[img_index].split()
        x_expected = int(gt_data_pos[0])
        y_expected = int(gt_data_pos[1])
        gt_patch = self.template[y_expected: y_expected + patch.shape[0], x_expected: x_expected + patch.shape[1]]

        self.curr_image_ind += 1

        return patch, gt_patch