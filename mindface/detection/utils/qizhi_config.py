import argparse
import math
import mindspore

from runner import read_yaml, TrainingWrapper

from mindspore.context import ParallelMode
import mindspore.ops as ops
import time
import moxing as mox
from mindspore.train.callback import Callback
import os
import sys

class UploadOutput(Callback):
    def __init__(self, train_dir, obs_train_url):
        self.train_dir = train_dir
        self.obs_train_url = obs_train_url
    def epoch_end(self,run_context):
        try:
            mox.file.copy_parallel(self.train_dir , self.obs_train_url )
            print("Successfully Upload {} to {}".format(self.train_dir ,self.obs_train_url ))
        except Exception as e:
            print('moxing upload {} to {} failed: '.format(self.train_dir ,self.obs_train_url ) + str(e))
        return  

### Copy single dataset from obs to training image###
def ObsToEnv(obs_data_url, data_dir):
    try:     
        mox.file.copy_parallel(obs_data_url, data_dir)
        print("Successfully Download {} to {}".format(obs_data_url, data_dir))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(obs_data_url, data_dir) + str(e))
    #Set a cache file to determine whether the data has been copied to obs. 
    #If this file exists during multi-card training, there is no need to copy the dataset multiple times.
    f = open("/cache/download_input.txt", 'w')    
    f.close()
    try:
        if os.path.exists("/cache/download_input.txt"):
            print("download_input succeed")
    except Exception as e:
        print("download_input failed")
    return 
### Copy the output to obs###
def EnvToObs(train_dir, obs_train_url):
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir,obs_train_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(train_dir,obs_train_url) + str(e))
    return      
def DownloadFromQizhi(obs_data_url, data_dir):
    device_num = int(os.getenv('RANK_SIZE'))
    if device_num == 1:
        ObsToEnv(obs_data_url,data_dir)
    #    context.set_context(mode=context.GRAPH_MODE,device_target=args.device_target)
    # if device_num > 1:
    #     # set device_id and init for multi-card training
    #     # context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=int(os.getenv('ASCEND_DEVICE_ID')))
    #     # context.reset_auto_parallel_context()
    #     # context.set_auto_parallel_context(device_num = device_num, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True, parameter_broadcast=True)
    #     # init()
    #     #Copying obs data does not need to be executed multiple times, just let the 0th card copy the data
    #     local_rank=int(os.getenv('RANK_ID'))
    #     if local_rank%8==0:
    #         ObsToEnv(obs_data_url,data_dir)
    #     #If the cache file does not exist, it means that the copy data has not been completed,
    #     #and Wait for 0th card to finish copying data
    #     while not os.path.exists("/cache/download_input.txt"):
    #         time.sleep(1)  
    # return
def UploadToQizhi(train_dir, obs_train_url):
    device_num = int(os.getenv('RANK_SIZE'))
    local_rank=int(os.getenv('RANK_ID'))
    if device_num == 1:
        EnvToObs(train_dir, obs_train_url)
    if device_num > 1:
        if local_rank%8==0:
            EnvToObs(train_dir, obs_train_url)
    return