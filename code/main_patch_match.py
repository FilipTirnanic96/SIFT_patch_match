import os
from PIL import Image

from patch_matcher.patch_matcher import PatchMatcher
from patch_matcher.simple_patch_matcher import SimplePatchMatcher
from utils.glob_def import DATA_DIR, DATA, REPORT_DIR
from patch_matcher.advance_patch_matcher import AdvancePatchMatcher
from kpi_calculation.calculate_kpi import CalculateKPI 
from pach_match_visualisation.match_visualisation import visualise_match, visualise_match2
import numpy as np
import pandas as pd


def get_patch_matcher(template: np.array, patch_matcher_type: str) -> PatchMatcher:
    """
    Returns patch matcher object from definer patch matcher type.

    :param template: Template image
    :param patch_matcher_type: Type of patch matcher
    :return: Patch matcher object
    """

    # get patch matcher
    if patch_matcher_type == 'simple':
        patch_matcher = SimplePatchMatcher(template)
    elif patch_matcher_type == 'advanced':
        patch_matcher = AdvancePatchMatcher(template)
    else:
        raise ValueError("Patch matcher type must be simple or advanced")

    return patch_matcher


def generate_kpi_reports(list_file_names: list, patch_matcher_type: str, model_name: str):
    """
    Generate KPI reports for input list file names.

    :param list_file_names: Input list of txt number file names
    :param patch_matcher_type: Type of patch matcher
    :param model_name: Model name for creating report
    """

    # get map template image
    template_image_path = os.path.join(DATA_DIR, "set", "map.png")
    template = Image.open(template_image_path)

    # get patch matcher
    patch_matcher = get_patch_matcher(template, patch_matcher_type)

    # generate KPI reports
    kpi = CalculateKPI(DATA_DIR, patch_matcher, model_name)
    kpi.calculate_kpis_from_inputs(list_file_names)


def visualise_patch_matches(kpi_df: pd.DataFrame, patch_matcher_type: str, num_to_visualise: int):
    """
    Visualise patches match with template.

    :param kpi_df: KPI report for patches
    :param patch_matcher_type: Type of patch matcher
    :param num_to_visualise: Number of random patches to be visualised
    """

    # get map template image
    template_image_path = os.path.join(DATA_DIR, "set", "map.png")
    template = Image.open(template_image_path)
    # get patch matcher
    patch_matcher = get_patch_matcher(template, patch_matcher_type)

    # random sample n patches
    df_kpi_sample = kpi_df.sample(num_to_visualise)

    # visualise matches
    visualise_match(template, patch_matcher, df_kpi_sample)


def visualise_patch_matches2(kpi_df: pd.DataFrame, patch_matcher_type: str, num_to_visualise: int):
    # get map template image
    template_image_path = os.path.join(DATA_DIR, "set", "map.png")
    template = Image.open(template_image_path)
    # get patch matcher
    patch_matcher = get_patch_matcher(template, patch_matcher_type)
    visualise_match2(template, patch_matcher, kpi_df)

    return


if __name__ == "__main__":
    """
    Test patch match implementation. 
    
    Flag 1 will generate report for file names defined in file_names_list.
    Set Data variable in utils/glob_def.py to "public" or "private" depending on which data 
    you want to use.
    
    Flag 2 will show how patch matcher matched random sampled patches for report provided with
    string path_to_report_csv. Number of random sampled patches is defined with num_to_visualise_.
    """

    test_implementation_flag = 1
    if test_implementation_flag == 1:
        file_names_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        patch_matcher_type_ = "advanced"
        model_name = patch_matcher_type_ + "_patch_matcher_" + DATA
        generate_kpi_reports(file_names_list, patch_matcher_type_, model_name)

    elif test_implementation_flag == 2:
        path_to_report_csv = r"\advance_patch_matcher_" + DATA + r"\4.txt_patch_matched.csv"
        df_kpi = pd.read_csv(REPORT_DIR + path_to_report_csv)
        patch_matcher_type_ = "simple"
        num_to_visualise_ = 30
        visualise_patch_matches(df_kpi, patch_matcher_type_, num_to_visualise_)

    elif test_implementation_flag == 3:
        path_to_report_csv = r"\advance_patch_matcher_" + DATA + r"\1.txt_patch_matched.csv"
        df_kpi = pd.read_csv(REPORT_DIR + path_to_report_csv)
        patch_matcher_type_ = "advanced"
        num_to_visualise_ = 30
        visualise_patch_matches2(df_kpi, patch_matcher_type_, num_to_visualise_)