# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:44:17 2022

@author: uic52421
"""
import os
from PIL import Image
import numpy as np
import pandas as pd
from patch_matcher.patch_matcher import PatchMatcher
from patch_matcher.simple_patch_matcher import SimplePatchMatcher
from kpi_calculation.report import Report


class CalculateKPI:

    def __init__(self, path_to_dataset: str, patch_matcher_: PatchMatcher, model_name: str = "adv_pm"):
        self.dataset_path = path_to_dataset
        self.path_to_input_txt_files = os.path.join(path_to_dataset, 'inputs')
        self.path_to_output_txt_files = os.path.join(path_to_dataset, 'outputs')
        self.path_to_input_patches = os.path.join(path_to_dataset, 'set')
        # init patch matcher
        self.patch_matcher_ = patch_matcher_
        # init report
        self.report = Report(model_name)

    def calculate_kpis_from_inputs(self, file_names_number: list):
        if len(file_names_number) == 0:
            return

        cumulative_matched = 0
        cumulative_num_patches = 0
        cumulative_time = 0
        for file_name_number in file_names_number:
            df_kpi = self.calculate_kpis(file_name_number)

            num_patches_to_process = df_kpi.shape[0]
            accuracy = (sum(df_kpi['matched'] == 1)) / num_patches_to_process
            time_taken = sum(df_kpi['time'])
            time_taken1 = sum(df_kpi['time_extract_kp'])
            time_taken2 = sum(df_kpi['time_extract_feat'])
            time_taken3 = sum(df_kpi['time_extract_match'])
            time_taken4 = sum(df_kpi['time_extract_loc'])

            print('Accuracy for filename', file_name_number, '.txt for n =', num_patches_to_process, 'processed patches is', accuracy)
            print('Time taken for filename', file_name_number, '.txt for n =', num_patches_to_process, 'processed patches is', time_taken)
            '''print('Time time_extract_kp taken for filename', file_name_number, '.txt for n =', num_patches_to_process,
                  'processed patches is', time_taken1)
            print('Time  time_extract_feat taken for filename', file_name_number, '.txt for n =', num_patches_to_process,
                  'processed patches is', time_taken2)
            print('Time time_extract_match taken for filename', file_name_number, '.txt for n =', num_patches_to_process,
                  'processed patches is', time_taken3)
            print('Time time_extract_loc taken for filename', file_name_number, '.txt for n =', num_patches_to_process,
                  'processed patches is', time_taken4)'''

            cumulative_matched += sum(df_kpi['matched'] == 1)
            cumulative_num_patches += num_patches_to_process
            cumulative_time += time_taken

        if len(file_names_number) > 1:
            cumulative_accuracy = cumulative_matched / cumulative_num_patches
            cumulative_time = cumulative_time / cumulative_num_patches
            print('Overall Accuracy for n =', cumulative_num_patches, 'processed patches is', cumulative_accuracy)
            print('Overall avg time taken for n =', cumulative_num_patches, 'processed patches is', cumulative_time)

        return

    def calculate_kpis(self, file_name_number: int = -1):
        # process template image
        template_image_path = os.path.join(self.dataset_path, "set", "map.png")
        template = Image.open(template_image_path)

        # init list to store kpis
        kpis = []

        # read each txt file
        for txt_file in os.listdir(self.path_to_input_txt_files):

            # calculate KPI just for specific input file
            if file_name_number != -1:
                txt_file_name_number = int(txt_file[0])
                if file_name_number != txt_file_name_number:
                    continue

            # open ground truth file
            output_f = open(os.path.join(self.path_to_output_txt_files, txt_file), 'r')

            # init kpi for current file
            kpis_file = []

            # open current input.txt file
            with open(os.path.join(self.path_to_input_txt_files, txt_file), 'r') as f:
                # read initial params
                path_to_template = f.readline()
                num_patches = int(f.readline())
                patch_size_str = f.readline().split()
                ph = int(patch_size_str[0])
                pw = int(patch_size_str[1])

                # if we have simple patch matcher we need to calculate new template features for different patch sizes
                if type(self.patch_matcher_) is SimplePatchMatcher:
                    self.patch_matcher_ = SimplePatchMatcher(template, pw, ph)

                # read path to each patch
                path_to_patch = f.readline()
                while path_to_patch:
                    # init kpi list for current patch 
                    kpi_list = []

                    # take relative path from dataset path
                    path_to_patch = path_to_patch.split('/')
                    path_to_patch = os.path.join(path_to_patch[1], path_to_patch[2])
                    # delete new line read from input
                    path_to_patch = path_to_patch[:-1]
                    # append relative to list
                    kpi_list.append(path_to_patch)
                    # make full path to patch
                    path_to_patch = os.path.join(self.path_to_input_patches, path_to_patch)

                    # read patch
                    patch = Image.open(path_to_patch)

                    # read expected output
                    expected_output = output_f.readline().split()
                    x_expected = int(expected_output[0])
                    y_expected = int(expected_output[1])
                    self.patch_matcher_.expected_x = x_expected
                    self.patch_matcher_.expected_y = y_expected

                    # match patch to template
                    x_match, y_match = self.patch_matcher_.match_patch(patch)

                    # append x_match, y_match to list
                    kpi_list.append(x_match)
                    kpi_list.append(y_match)

                    # append x_expected, y_expected to list
                    kpi_list.append(x_expected)
                    kpi_list.append(y_expected)

                    # append num of matched points
                    kpi_list.append(self.patch_matcher_.n_points_matched)

                    # check if we are in 120 neighborhood (6x6 pixels missed)
                    matched = False
                    if abs(x_expected - x_match) <= 20 and abs(y_expected - y_match) <= 20:
                        matched = True

                    # append matched to list
                    kpi_list.append(matched)

                    # append time taken to list
                    kpi_list.append(self.patch_matcher_.time_passed_sec)
                    kpi_list.append(self.patch_matcher_.time_passed_sec_extract_kp)
                    kpi_list.append(self.patch_matcher_.time_passed_sec_extract_feat)
                    kpi_list.append(self.patch_matcher_.time_passed_sec_match)
                    kpi_list.append(self.patch_matcher_.time_passed_sec_loc)
                    # store kpi_list to list of all kpis
                    kpis.append(kpi_list)

                    # store kpi_list to list of current file kpis
                    kpis_file.append(kpi_list)

                    # read new patch path
                    path_to_patch = f.readline()

            df_kpi_file = pd.DataFrame(kpis_file,
                                  columns=['path', 'x_match', 'y_match', 'x_expected', 'y_expected', 'n_points_matched',
                                           'matched', 'time', 'time_extract_kp', 'time_extract_feat', 'time_extract_match', 'time_extract_loc'])

            self.report.make_report(df_kpi_file, txt_file)
            # close current output file
            output_f.close()

        df_kpi = pd.DataFrame(kpis, columns=['path', 'x_match', 'y_match', 'x_expected', 'y_expected', 'n_points_matched',
                                       'matched', 'time', 'time_extract_kp', 'time_extract_feat', 'time_extract_match', 'time_extract_loc'])

        if file_name_number == -1:
            self.report.make_report(df_kpi, "all_data.txt")

        # return report
        return df_kpi
