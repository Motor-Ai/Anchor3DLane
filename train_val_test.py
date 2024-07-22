import ast
import os 
import argparse
from typing import Tuple
from collections import Counter
from sklearn.model_selection import train_test_split

import pandas as pd

def group_zod_metadata_into_train_val_test(df: pd.DataFrame, stratify_col: str, 
                                           test_split_countries: list, train_val_split_size: float = 0.2, save_as_txt: bool = False) \
                                                                                -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training, validation, and test sets based on country codes and a stratification column.

    :param df: DataFrame containing the data
    :param stratify_col: The column name to stratify the train/validation split
    :param test_split_countries: List of country codes to be included in the test set
    :param train_val_split_size: Proportion of the train/validation set to be used as the validation set
    :return: A tuple containing three DataFrames: (train, validation, test)
    """
    
    # Create a DataFrame with only the specified test split countries
    df_test = df[df['country_code'].isin(test_split_countries)]

    # Create a DataFrame without the specified test split countries
    df_train_val = df[~df['country_code'].isin(test_split_countries)]
    
    # Split the train/validation DataFrame into training and validation sets, stratified by the specified column
    df_train, df_val = train_test_split(df_train_val, test_size=train_val_split_size, stratify=df_train_val[stratify_col], random_state=42)
    
    return df_train, df_val, df_test

def find_folders_with_3d_anno(root_dir, file_name):
    folders_with_file = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if file_name in filenames:
            folders_with_file.append(int(os.path.relpath(dirpath, root_dir)[:6]))
    
    return folders_with_file

def format_number(number, length=6):
    return f"{number:0{length}d}"

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataframe_path', type=str, default='metadata_dataframe.csv', help='the path to the csc dataframe')
    parser.add_argument('--root_dir', type=str, default='data/zod_dataset/single_frames', help='the path to the csc dataframe')
    args = parser.parse_args()
    return args

  

if __name__ == "__main__":
    args = parse_config()
    
    # the folders where there was no instance id do not have ane annotations, fo first we need to find which folders can be used for training 
    folders_with_anno = find_folders_with_3d_anno(args.root_dir, 'lane_markings_3d_ol_style.json')
    
    df = pd.read_csv(args.dataframe_path)
    # less than 10 percent countries ['FR', 'NO', 'HU', 'GB', 'IE', 'NL', 'CZ', 'LU', 'FI', 'SK']
    percentages = df['country_code'].value_counts(normalize=True) * 100
    test_country_codes = percentages[percentages <= 10].index
    
    filtered_df = df[df['frame_id'].isin(folders_with_anno)]
    df_train, df_val, df_test = group_zod_metadata_into_train_val_test(filtered_df, 'time_of_day', test_country_codes)
    
    train_list, val_list, test_list = list(df_train['frame_id']), list(df_val['frame_id']), list(df_test['frame_id'])
    
    with open('data/zod_dataset/data_lists/full_training.txt', 'w') as f:
        for one_path in train_list:
            f.write(f"{format_number(one_path)}\n")
        
    with open('data/zod_dataset/data_lists/full_validation.txt', 'w') as f:
        for one_path in val_list:
            f.write(f"{format_number(one_path)}\n")
        
    with open('data/zod_dataset/data_lists/full_test.txt', 'w') as f:
        for one_path in test_list:
            f.write(f"{format_number(one_path)}\n")
    
    