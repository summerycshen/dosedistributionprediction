import os, sys, pdb, random, math, datetime, shutil
from scipy import io
import numpy as np
import torch,cv2
import torch.nn.parallel
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloaders.dose_loader_rectum_3d import Dose_online_avoid_test
import dataloaders.dose_transforms_3d as tr
from networks_cfgs.deeplab import deeplab_res50_cfg
import tasks.rectum_tasks as rectum_tasks
from networks.res_unet import *

save_dir = 'mat_save/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


class Config(object):
    def __init__(self):
        self.train_batch = 3
        self.validation_batch = 2
        self.dataset = "rectum"
        self.data_root = "./data/rectumtest"
        # self.test_txt = "./datalist/mini_train.txt"
        # self.test_txt = "./datalist/test.txt"
        self.test_txt = "./datalist/test.txt"

        self.nepoch = 500
        self.HU_max = 390
        self.HU_min = -310
        self.prescription = 46.8

        self.mask_dict = rectum_tasks.mask_dict_5OARS  # mask_dict_4OARS | mask_dict_all
        self.s_h = 224  # target height
        self.s_w = 224
        # self.rotation = 15
        self.rotation = 0

        self.optimizer = "adam"
        self.lr = 3e-4  # 1.0*1e-7
        self.in_channels = len(self.mask_dict) + 1  # 3 | len(self.mask_dict)
        self.num_classes = 1
        self.wd = 5e-4
        self.momentum = 0.9

        self.network = 'res_unet_v6'
        self.net_config = deeplab_res50_cfg  # res_unet50_regularize_cfg([0.5, 0.5, 0.5])

        self.suffix = "vanilla"  # inception_perception_sgd_aug
        # self.checkpoint = "./checkpoint/rectum/res_unet/vanilla/fuison.pth"
        self.checkpoint = "./checkpoint/rectum/res_unet_v6/epoch_50.pth"
        self.gpus = "0"
        self.num_workers = 2
        self.manualSeed = None


if __name__ =="__main__":
    config = Config()
    base_tr1 = transforms.Compose([
            tr.FilterHU(config.HU_min, config.HU_max),
            tr.NormalizeCT(config.HU_min, config.HU_max),
            tr.Arr2image(),
            tr.AlignCT(),
            tr.Padding([config.s_h, config.s_w]),  # h, w
    ])
    base_tr2 = transforms.Compose([
            tr.NormalizeDosePerSample(),
            tr.Stack2Tensor()
    ])
    base_transformer = {'step1':base_tr1, 'step2':base_tr2}

    sample_dict_list = []
    fp = open(config.test_txt, "r")
    lines = fp.readlines()
    fp.close()
    for line in lines:
        sample_masks = {}
        line = line.strip()
        if line != "" and not line.startswith("#"):
            sample_name = line.split("\t")[0]
            mask_names = line.split("\t")[1:]
            sample_masks[sample_name] = mask_names
            sample_dict_list.append(sample_masks)

    validationset = Dose_online_avoid_test(config.data_root,
                                    sample_txt=config.test_txt,
                                    mask_dict=config.mask_dict,
                                    base_transforms = base_transformer,
                                    transforms=None,
                                    remake=True)

    validationset_loader = DataLoader(validationset,
                                      batch_size=1,
                                      shuffle=False,
                                        num_workers=0)

    # model = ResUNetPlus()
    if "res_unet_v3" in config.network:
        model = ResUNet_v3()
    elif "res_unet_v4" in config.network:
        model = ResUNet_v4()
    elif "res_unet_v5" in config.network:
        model = ResUNet_v5()
    elif "res_unet_v6" in config.network:
        model = ResUNet_v6()
    elif "res_unet_v7" in config.network:
        model = ResUNet_v7()
    elif "res_unet_v8" in config.network:
        model = ResUNet_v8()
    elif "res_unet_v10" in config.network:
        model = ResUNet_v10()
    elif "res_unet_v12" in config.network:
        model = ResUNet_v12()
    else:
        TypeError("unsupport network......")

    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)

    device = torch.device("cpu")
    model.to(device)

    os.makedirs("./result", exist_ok=True)
    os.makedirs("./result1", exist_ok=True)
    model.eval()
    with torch.no_grad():
        print(len(validationset_loader))
        for batch_idx, (sample_batched, case_id) in enumerate(validationset_loader):
            max_slices = sample_batched[-1][2].item()
            all_predict = torch.zeros((max_slices, 224, 224))
            all_label = torch.zeros((max_slices, 224, 224))

            for patch_list in sample_batched:
                [patch, j, k] = patch_list
                inputs, labels = patch['input'], patch['rd_slice']
                inputs = inputs.to(device).float()
                labels = labels.float()
                outputs = model(inputs)
                outputs = torch.squeeze(outputs, 1).cpu()
                all_predict[j:k, ...] = outputs
                all_label[j:k, ...] = labels

            all_predict = all_predict.numpy().transpose(1, 2, 0)
            all_label = all_label.numpy().transpose(1, 2, 0)
            compare = np.zeros((all_label.shape[0], all_label.shape[1] * 2), dtype=np.uint8)
            for i in range(all_label.shape[2]):
                compare[:, :all_label.shape[1]] = (all_label[:, :, i] * 255).astype(np.uint8)
                compare[:, all_label.shape[1]:] = (all_predict[:, :, i] * 255).astype(np.uint8)
                cv2.imwrite('result1/' + str(i) + '.png', compare)
            data = {'label_D95': all_label, 'prediction_D95': all_predict}
            print(case_id)
            save_path = os.path.join(save_dir, case_id[0] + '.mat')
            io.savemat(save_path, data)

