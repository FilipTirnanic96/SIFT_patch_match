import os.path
import shutil

import pandas as pd

from utils.glob_def import REPORT_DIR, CONFIG_DIR
import matplotlib.pyplot as plt
import matplotlib


class Report:

    def __init__(self, model_name: str = "advance_pm"):
        # set directory where reports will be saved
        self.reports_folder_path = REPORT_DIR
        # set name of used patch match model
        self.model_name = model_name

    def make_report(self, df: pd.DataFrame, input_name: str):
        """
        Makes report of patch matcher KPI results.
        Saves figures and cvs files.

        :param df: Patch matcher KPI results
        :param input_name: Name of input txt file
        """

        # make folder for results
        output_dir = os.path.join(self.reports_folder_path, self.model_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # set matplotlib style
        matplotlib.style.use('ggplot')
        # bar plot percent of true detections
        detection_statistics = df['matched'].value_counts() / df.shape[0]
        plt.figure(figsize=(10, 8))
        plt.title("Percentage of miss detection of " + str(df.shape[0]) + " patches")

        ax = detection_statistics.plot.bar()
        ax.bar_label(ax.containers[0])

        plt.savefig(os.path.join(output_dir, input_name + '_det_stat.png'))

        # get all matched patches
        # bar plot number of histogram of n_points_matched for matched patches
        patch_matched_df = df[df['matched'] == True]
        if patch_matched_df.shape[0] > 0:
            number_of_n_points_matched = patch_matched_df['n_points_matched'].value_counts()
            number_of_n_points_matched = number_of_n_points_matched.sort_index()
            plt.figure(figsize=(10, 8))
            plt.title("Number of matched points in " + str(patch_matched_df.shape[0]) + " data")
            ax = number_of_n_points_matched.plot.bar()
            ax.bar_label(ax.containers[0])

            plt.savefig(os.path.join(output_dir, input_name + '_n_points_matched.png'))

            # save matched patch details
            patch_matched_df.to_csv(os.path.join(output_dir, input_name + '_patch_matched.csv'))

        # get all miss matched patches
        # bar plot number of histogram of n_points_matched for miss matched patches
        patch_miss_matched_df = df[df['matched'] == False]
        if patch_miss_matched_df.shape[0] > 0:
            number_of_n_points_matched = patch_miss_matched_df['n_points_matched'].value_counts()
            number_of_n_points_matched = number_of_n_points_matched.sort_index()
            plt.figure(figsize=(10, 8))
            plt.title("Number of matched points in " + str(patch_miss_matched_df.shape[0]) + " miss detections")
            ax = number_of_n_points_matched.plot.bar()
            ax.bar_label(ax.containers[0])

            plt.savefig(os.path.join(output_dir, input_name + '_n_points_matched_miss.png'))

            # save miss matched patch details
            patch_miss_matched_df.to_csv(os.path.join(output_dir, input_name + '_patch_miss_matched.csv'))

        # save algo parameters
        shutil.copyfile(os.path.join(CONFIG_DIR, "patch_match_cfg.yml"),
                        os.path.join(output_dir, "patch_match_cfg.yml"))

        plt.close()
