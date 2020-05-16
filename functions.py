import os
import re
from scipy.io import wavfile
import glob 
import pandas as pd 
import textgrids
import numpy as np


#functions for getting data from original folder with all of the audio data 
def get_wav_files(folder_name='original_en_diapix_data'):
    cwd = os.getcwd()
    correct_folder = cwd + "/" + folder_name
    return glob.glob(correct_folder + "/*.wav")

def get_file_names(file_name):
    a = re.findall(r"(ENF?_\d\d)", file_name)
    channel1 = a[0]
    if a[1] != a[0]:
        channel2 = a[1]
    return channel1, channel2

# def create_directory_move_file(file_to_be_moved, subdirectory_name = "subdirectory_name"):
#     cwd = os.getcwd()
#     os.mkdir(subdirectory_name)
#     for item in file_to_be_moved:
#         os.replace(cwd + '/' + item, cwd + '/' + subdirectory_name + '/' + item)
        
def split_save_wav(wav_file_list):
    cwd = os.getcwd()
    for item in wav_file_list:
        fs, data = wavfile.read(item)
        channel1, channel2 = get_file_names(item)
        wavfile.write(channel1, fs, data[:, 0])   # saving first column which corresponds to channel 1
        wavfile.write(channel2, fs, data[:, 1])   # saving second column which corresponds to channel 2
        os.replace(cwd + '/' + channel1, cwd + '/' + "split_wav_files_folder" + '/' + channel1)
        os.replace(cwd + '/' + channel2, cwd + '/' + "split_wav_files_folder" + '/' + channel2)
        
def get_textgrids_for_each_speaker(folder = "/original_en_diapix_data_changed_textgrids"):
        cwd = os.getcwd()
        textgrid_directory = cwd + folder
        textgrid_list = glob.glob(textgrid_directory + "/*.TextGrid")
        return textgrid_list
    
def make_mixed_into_lists(mixed_df):
    text = []
    xmin = []
    xmax = []
    for item in mixed_df[0]:
        if re.match(r"<Interval\stext=\"(.*)\"\sxmin=(.+)\sxmax=(.+)>", str(item)) != None:
            x = re.match(r"<Interval\stext=\"(.*)\"\sxmin=(.+)\sxmax=(.+)>" , str(item))
            if x.group(1) == "":
                text.append(np.nan)
            else:
                text.append(x.group(1))
            xmin.append(float(x.group(2)))
            xmax.append(float(x.group(3)))
        else:
            raise Exception
    mixed_df['Word_Text'] = text
    mixed_df['Word_xmin'] = xmin
    mixed_df['Word_xmax'] = xmax
    mixed_df.drop(columns = [0], inplace=True)
    return mixed_df

def make_phone_into_lists(phone_df):
    text = []
    xmin = []
    xmax = []
    for item in phone_df[0]:
        if re.match(r"<Interval\stext=\"(.*)\"\sxmin=(.+)\sxmax=(.+)>", str(item)) != None:
            x = re.match(r"<Interval\stext=\"(.*)\"\sxmin=(.+)\sxmax=(.+)>", str(item))
            if x.group(1) == "":
                text.append(np.nan)
            else:
                text.append(x.group(1))
            xmin.append(float(x.group(2)))
            xmax.append(float(x.group(3)))
        else:
            raise Exception
    phone_df['Phone_Text'] = text
    phone_df['Phone_xmin'] = xmin
    phone_df['Phone_xmax'] = xmax
    phone_df.drop(columns = [0], inplace=True)
    return phone_df

def combine_dfs(grid):
    # channel1
    # make into a df
    grid_mixed_df = pd.DataFrame.from_dict(grid['mixed'])
    grid_phone_df = pd.DataFrame.from_dict(grid['phone'])
    # making the df have columns
    mixed_df = make_mixed_into_lists(grid_mixed_df)
    phone_df = make_phone_into_lists(grid_phone_df)
    # combining the two dataframes 
    combined_df = phone_df.merge(mixed_df, how='left', left_on='Phone_xmin', right_on='Word_xmin')
    combined_df.fillna(method = 'ffill', inplace=True)
    
    #channel 2
    grid_mixed_df2 = pd.DataFrame.from_dict(grid['mixed2'])
    grid_phone_df2 = pd.DataFrame.from_dict(grid['phone2'])
    # making the df have columns
    mixed_df2 = make_mixed_into_lists(grid_mixed_df2)
    phone_df2 = make_phone_into_lists(grid_phone_df2)
    # combining the two dataframes 
    combined_df2 = phone_df2.merge(mixed_df2, how='left', left_on='Phone_xmin', right_on='Word_xmin')
    combined_df2.fillna(method = 'ffill', inplace=True)
    return combined_df, combined_df2

def split_and_name_textgrids(original_folder = "/original_en_diapix_data_changed_textgrids", 
                             destination_folder = '/split_wav_files_folder/'):
    cwd = os.getcwd()
    textgrid_list = get_textgrids_for_each_speaker(folder = original_folder)
    for file_name in textgrid_list:
        channel1, channel2 = get_file_names(file_name)
        grid = textgrids.TextGrid(file_name)
        chan1, chan2 = combine_dfs(grid)
        # save to csv
        chan1.to_csv(cwd + '/' + destination_folder + '/' + channel1 + ".TextGrid")
        chan2.to_csv(cwd + '/' + destination_folder + '/' + channel2 + ".TextGrid")

# functions for adding to the dataframes

# functions for downsampling data

# functions for hampel filter 
        
      