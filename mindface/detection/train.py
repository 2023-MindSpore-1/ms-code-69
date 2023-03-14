# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Train Retinaface_ResNet50 or Retinaface_MobileNet0.25."""

import argparse
import math
import mindspore

from mindspore import context
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from loss import MultiBoxLoss
from datasets import create_dataset
from utils import adjust_learning_rate
from utils.qizhi_config import *

from models import RetinaFace, RetinaFaceWithLossCell, resnet50, mobilenet025
from runner import read_yaml, TrainingWrapper

def train(cfg,args):
    """train"""
    mindspore.common.seed.set_seed(cfg['seed'])

    if cfg['mode'] == 'Graph':
        context.set_context(mode=context.GRAPH_MODE, device_target=cfg['device_target'])
    else :
        context.set_context(mode=context.PYNATIVE_MODE, device_target = cfg['device_target'])

    # rank=0
    if cfg['device_target'] == "Ascend":
        device_num = cfg['nnpu']
        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
            rank = get_rank()
            print(f"The rank ID of current device is {rank}.")
        else:
            context.set_context(device_id=cfg['device_id'])
    elif cfg['device_target'] == "GPU":
        if cfg['ngpu'] > 1:
            init("nccl")
            context.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            rank = get_rank()
            print(f"The rank ID of current device is {rank}.")



    batch_size = cfg['batch_size']
    max_epoch = cfg['epoch']
    clip = cfg['clip']
    momentum = cfg['momentum']
    lr_type = cfg['lr_type']
    weight_decay = cfg['weight_decay']
    initial_lr = cfg['initial_lr']
    gamma = cfg['gamma']
    training_dataset = args.local_path + cfg['training_dataset']
    num_classes = cfg['num_classes']
    negative_ratio = 7
    stepvalues = (cfg['decay1'], cfg['decay2'])

    ds_train = create_dataset(training_dataset, cfg['variance'], cfg['match_thresh'], cfg['image_size'],
                                clip, batch_size, multiprocessing=True, num_worker=cfg['num_workers'])
    print('dataset size is : \n', ds_train.get_dataset_size())

    steps_per_epoch = math.ceil(ds_train.get_dataset_size())

    multibox_loss = MultiBoxLoss(num_classes, cfg['num_anchor'], negative_ratio)
    if cfg['name'] == 'ResNet50':
        backbone = resnet50(1001)
    elif cfg['name'] == 'MobileNet025':
        backbone = mobilenet025(1000)
    backbone.set_train(True)

    if  cfg['pretrain'] and cfg['resume_net'] is None:
        DownloadFromQizhi(args.ckpt_url, data_dir=local_path+'/'+args.ckpt_url.split('/')[-1])
        pretrained= args.local_path+'/'+args.ckpt_url.split('/')[-1]
        print(pretrained)
        param_dict = load_checkpoint(pretrained)
        load_param_into_net(backbone, param_dict)
        print(f"Load RetinaFace_{cfg['name']} from [{cfg['pretrain_path']}] done.")

    net = RetinaFace(phase='train', backbone=backbone, in_channel=cfg['in_channel'], out_channel=cfg['out_channel'])
    net.set_train(True)

    if cfg['resume_net'] is not None:
        pretrain_model_path = cfg['resume_net']
        param_dict_retinaface = load_checkpoint(pretrain_model_path)
        load_param_into_net(net, param_dict_retinaface)
        print(f"Resume Model from [{cfg['resume_net']}] Done.")

    loc_weight = cfg['loc_weight']
    class_weight = cfg['class_weight']
    landm_weight = cfg['landm_weight']
    net = RetinaFaceWithLossCell(net, multibox_loss, loc_weight, class_weight, landm_weight)

    lr = adjust_learning_rate(initial_lr, gamma, stepvalues, steps_per_epoch, max_epoch,
                              warmup_epoch=cfg['warmup_epoch'], lr_type1=lr_type)

    if cfg['optim'] == 'momentum':
        opt = mindspore.nn.Momentum(net.trainable_params(), lr, momentum,weight_decay, loss_scale=1)
    elif cfg['optim'] == 'sgd':
        opt = mindspore.nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=momentum,
                               weight_decay=weight_decay, loss_scale=1)
    else:
        raise ValueError('optim is not define.')

    net = TrainingWrapper(net, opt, grad_clip=cfg['grad_clip'])

    model = Model(net)

    config_ck = CheckpointConfig(save_checkpoint_steps=cfg['save_checkpoint_steps'],
                                 keep_checkpoint_max=cfg['keep_checkpoint_max'])
    ckpoint_cb = ModelCheckpoint(prefix="RetinaFace", directory=cfg['ckpt_path'], config=config_ck)

    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    callback_list = [LossMonitor(), time_cb, ckpoint_cb]

    print("============== Starting Training ==============")
    model.train(max_epoch, ds_train, callbacks=callback_list, dataset_sink_mode=False)

if __name__ == '__main__':

    local_path = os.path.abspath(__file__)[:-9]
    

    parser = argparse.ArgumentParser(description='train')
    # configs
    parser.add_argument('--data_url',
                    help='path to training/inference dataset folder',
                    default= '/cache/data/')

    parser.add_argument('--train_url',
                    help='output folder to save/load',
                    default= '/cache/output/')

    parser.add_argument('--ckpt_url', type=str, default='None',
                        help='checpoint path')


    parser.add_argument(
        '--device_target',
        type=str,
        default="Ascend",
        choices=['Ascend', 'CPU'],
        help='device where the code will be implemented (default: Ascend),if to use the CPU on the Qizhi platform:device_target=CPU')


    parser.add_argument('--config', default='RetinaFace_mobilenet025.yaml', type=str ,help='config path') 

    parser.add_argument('--local_path', help='local_path', default= local_path) 

    
    
    args = parser.parse_args()

    print(args.local_path)
    print(args.config)
    data_dir = '/cache/data'  
    train_dir = '/cache/output'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    ###Initialize and copy data to training image
    DownloadFromQizhi(args.data_url, data_dir=args.local_path+'/data')
    ###The dataset path is used here:data_dir +"/train"   


    config = read_yaml(local_path + '/configs/' + args.config)
    train(cfg=config,args =args)

    UploadToQizhi(train_dir,args.train_url)