import os
import re
from scipy.io import wavfile
import glob 



#functions for getting data from original folder with all of the audio data 
def get_wav_files():
    cwd = os.getcwd()
    correct_folder = cwd + '/original_en_diapix_data'
    return glob.glob(correct_folder + "/*.wav")

def get_file_names(file_name):
    a = re.findall(r"(ENF?_\d\d)", file_name)
    channel1 = a[0]
    if a[1] != a[0]:
        channel2 = a[1]
    return channel1, channel2

def create_directory_move_file(file_to_be_moved, subdirectory_name = "subdirectory_name"):
    cwd = os.getcwd()
    os.mkdir(subdirectory_name)
    for item in file_to_be_moved:
        os.replace(cwd + '/' + item, cwd + '/' + subdirectory_name + '/' + item)
        
def split_save_wav(wav_file_list):
    for item in wav_file_list:
        fs, data = wavfile.read(item)
        channel1, channel2 = get_file_names(item)
        wavfile.write(channel1, fs, data[:, 0])   # saving first column which corresponds to channel 1
        wavfile.write(channel2, fs, data[:, 1])   # saving second column which corresponds to channel 2
        os.replace(cwd + '/' + channel1, cwd + '/' + "split_wav_files_folder" + '/' + channel1)
        os.replace(cwd + '/' + channel2, cwd + '/' + "split_wav_files_folder" + '/' + channel2)
        
def get_textgrids_for_each_speaker():
        cwd = os.getcwd()
        textgrid_directory = cwd + "/original_en_diapix_data_changed_textgrids"
        textgrid_list = glob.glob(textgrid_directory + "/*.TextGrid")
        return textgrid_list