# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:44:17 2022

@author: uic52421
"""
import os
from PIL import Image
import pandas as pd
import numpy as np
from patch_matcher.patch_matcher import PatchMatcher
from kpi_calculation.report import Report


class CalculateKPI:

    def __init__(self, path_to_dataset: str, patch_matcher_: PatchMatcher, model_name: str = "adv_pm"):
        # init paths
        self.dataset_path = path_to_dataset
        self.path_to_input_txt_files = os.path.join(path_to_dataset, 'inputs')
        self.path_to_output_txt_files = os.path.join(path_to_dataset, 'outputs')
        self.path_to_input_patches = os.path.join(path_to_dataset, 'set')
        # init patch matcher
        self.patch_matcher_ = patch_matcher_
        # init report
        self.report = Report(model_name)
        # init columns of dataframe
        self.df_columns = ['path', 'x_match', 'y_match', 'x_expected', 'y_expected', 'n_points_matched',
                           'matched', 'time']

    def calculate_kpis_from_inputs(self, file_names_number: list):
        """
        Calculates KPI for giver list of file number as input. Generates report from
        dataframe of each filename input.

        :param file_names_number: List of file names used as input.
        """

        if len(file_names_number) == 0:
            return

        cumulative_matched = 0
        cumulative_num_patches = 0
        cumulative_time = 0
        over_all_df_kpi = pd.DataFrame([], columns = self.df_columns)
        # loop through all inputs
        for file_name_number in file_names_number:
            # calculate kpi for each input
            df_kpi = self.calculate_kpis(file_name_number)

            num_patches_to_process = df_kpi.shape[0]
            accuracy = (sum(df_kpi['matched'] == 1)) / num_patches_to_process
            time_taken = sum(df_kpi['time'])
            # print overall statistics
            print('Accuracy for filename', file_name_number, '.txt for n =', num_patches_to_process,
                  'processed patches is', accuracy)
            print('Time taken for filename', file_name_number, '.txt for n =', num_patches_to_process,
                  'processed patches is', time_taken)

            cumulative_matched += sum(df_kpi['matched'] == 1)
            cumulative_num_patches += num_patches_to_process
            cumulative_time += time_taken

            over_all_df_kpi = pd.concat([over_all_df_kpi, df_kpi], ignore_index=True)

        if len(file_names_number) > 1:
            self.report.make_report(over_all_df_kpi, "all_data.txt")
            cumulative_accuracy = cumulative_matched / cumulative_num_patches
            cumulative_time = cumulative_time / cumulative_num_patches
            # save overall statistics
            input_stat_df = pd.DataFrame(np.array([[0, 0]]), columns=['accuracy', 'time_taken'])
            input_stat_df['accuracy'] = cumulative_accuracy
            input_stat_df['time_taken'] = cumulative_time
            output_dir = os.path.join(self.report.reports_folder_path, self.report.model_name)
            input_stat_df.to_csv(os.path.join(output_dir, 'overall_model_statistics.csv'))
            # print overall statistics
            print('Overall Accuracy for n =', cumulative_num_patches, 'processed patches is', cumulative_accuracy)
            print('Overall avg time taken for n =', cumulative_num_patches, 'processed patches is', cumulative_time)

    def calculate_kpis(self, file_name_number: int = -1) -> pd.DataFrame:
        """
        Return dataframe for one input file and generates report from dataframe.
        Each data frame is array of each patch match statistics (ground truth
        patch position, estimated patch position, flag if match was successful
        and time match took). Match is successful if estimated patch position
        is around 20 x 20 pixels of true patch position.
        :param file_name_number: Number of file used as input
        :return: dataframe: Input statistics for each patch
        """

        # if file name is -1 return empty dataframe
        if file_name_number == -1:
            return pd.DataFrame([], columns = self.df_columns)

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
                # read path to template
                f.readline()
                # read num patches
                f.readline()
                # read initial params
                f.readline().split()

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

                    # check if we are in 20 neighborhood (20x20 area around expected patch position)
                    matched = False
                    if abs(x_expected - x_match) <= 20 and abs(y_expected - y_match) <= 20:
                        matched = True

                    # append matched to list
                    kpi_list.append(matched)

                    # append time taken to list
                    kpi_list.append(self.patch_matcher_.time_passed_sec)

                    # store kpi_list to list of all kpis
                    kpis.append(kpi_list)

                    # store kpi_list to list of current file kpis
                    kpis_file.append(kpi_list)

                    # read new patch path
                    path_to_patch = f.readline()

            # create dataframe
            df_kpi_file = pd.DataFrame(kpis_file, columns = self.df_columns)

            # make report from dataframe
            self.report.make_report(df_kpi_file, txt_file)
            # close current output file
            output_f.close()

        # crata
        df_kpi = pd.DataFrame(kpis, columns = self.df_columns)

        if file_name_number == -1:
            # make report for all data
            self.report.make_report(df_kpi, "all_data.txt")

        # return report
        return df_kpi
