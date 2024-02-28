from __future__ import print_function, division
import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve
from models.modeling_selection import SELECTION, CONFIGS
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from lifelines.utils import concordance_index

import argparse
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


tk_lim = 40

disease_list = ['OS','Non-OS']

parser = argparse.ArgumentParser(description="Trainning SELECTION for HAIC/TACE Treatment Decision")
parser.add_argument('--CLS', action='store', dest='CLS', type=int,help='No of classes')
parser.add_argument('--BSZ', action='store', dest='BSZ', type=int,help='Batch Size')
parser.add_argument('--epochs', action='store', dest='epochs', type=int,help='Epoch Size')
parser.add_argument('--lr', action='store', dest='lr', type=int,help='Learning Rate')
parser.add_argument('--seed', action='store', dest='seed', type=int,default=555,help='Trainning seed')
parser.add_argument('--EXP', action='store', dest='EXP', type=str',help='TACE/HAIC/TC/HC')
parser.add_argument('--EF', action='store', dest='EF', type=str,help='val/train/test')
parser.add_argument('--DATA_DIR', action='store', dest='DATA_DIR', type=str,
                    default= 'C:\\...'
                    ,help='Folder Path for images')
parser.add_argument('--SET_TYPE', action='store', dest='SET_TYPE', type=str,
                    default='D:\\...',
                    help= 'Folder path for clinical baseline data')
args = parser.parse_args()

def load_weights(model, weight_path):
    pretrained_weights = torch.load(weight_path, map_location=torch.device('cpu'))
    model_weights = model.state_dict()

    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}

    model_weights.update(load_weights)

    model.load_state_dict(model_weights)
    print("Loading SELECTION...")
    return model

def set_random_seed(seed=0, deterministic=True):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Data(Dataset):
    def __init__(self, set_type, img_dir,PathCat,transform=None, target_transform=None):
        dict_path = set_type+'.pkl'
        f = open(dict_path, 'rb')
        # self.mm_data = pickle.load(f)
        self.mm_data = pd.read_pickle(f)
        f.close()
        art_img_dir = os.path.join(img_dir,'arterial_SYSUCC_HAIC',PathCat)
        port_img_dir = os.path.join(img_dir, 'portal_SYSUCC_HAIC',PathCat)
        art_dirlist = os.listdir(art_img_dir)[0:]
        self.art_img_dir = art_img_dir
        self.port_img_dir = port_img_dir
        self.idx_list = art_dirlist
        self.art_dirlist = art_dirlist
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):

        return len(self.idx_list)
        # return len(self.X_data)

    def __getitem__(self, idx):
        k = int(self.idx_list[idx].replace('.npy',''))

        art_img_path = os.path.join(self.art_img_dir, str(k)) + '.npy'
        port_img_path = os.path.join(self.port_img_dir, str(k)) + '.npy'

        X1 = np.squeeze(np.load(art_img_path)).astype('float')
        X_m_1 = np.expand_dims(X1, axis=-1)
        art_img = np.concatenate([X_m_1, X_m_1, X_m_1], axis=-1)

        X2 = np.squeeze(np.load(port_img_path))
        X_m_2 = np.expand_dims(X2, axis=-1).astype('float')
        port_img = np.concatenate([X_m_2, X_m_2, X_m_2], axis=-1)


        label = self.mm_data[k]['label']    ## OOR
        label2 = self.mm_data[k]['label2']      # 1Y-OS status
        label3 = self.mm_data[k]['label3']*30       # OS time
        label = np.array([label, label2,label3]).astype(float)

        if self.transform:
            art_img = self.transform(art_img)
            port_img = self.transform(port_img)
        if self.target_transform:
            label = self.target_transform(label)

        demo = torch.as_tensor(np.array([self.mm_data[k]['age'],self.mm_data[k]['sex'],self.mm_data[k]['ECOG'],
                                         self.mm_data[k]['comorbidity'],self.mm_data[k]['hepatitis'],
                                         self.mm_data[k]['ascites']])).float()
        IV = torch.as_tensor(np.array([self.mm_data[k]['AFP'], self.mm_data[k]['tumour_n'], self.mm_data[k]['MaximumRadius'],
                      self.mm_data[k]['BCLC '], self.mm_data[k]['metastasis'],
                      self.mm_data[k]['PVTT']])).float()
        lab = torch.as_tensor(np.array([self.mm_data[k]['ALB'],self.mm_data[k]['ALT'],self.mm_data[k]['AST'],
                                         self.mm_data[k]['TBIL'],self.mm_data[k]['PT'],self.mm_data[k]['INR'],
                                         self.mm_data[k]['PLT'],self.mm_data[k]['child-pugh (5 or 6 or 7)']])).float()

        return art_img.to(torch.float32),port_img.to(torch.float32), label, demo, lab , IV,k

def test(args,load_model=None):
    torch.manual_seed(args.seed)
    num_classes = args.CLS
    EXP = args.EXP
    EXP_folder = args.EF
    config = CONFIGS["SELECTION"] # Get default selection configuration from the code.

    model = SELECTION(config, 224, zero_head=True, num_classes=num_classes)
    model = torch.nn.DataParallel(model)
    selection = load_weights(model, 'final_model\\'+EXP+'.pth')

    img_dir = os.path.join(args.DATA_DIR,'Model_test')

    ID= []
    score = []
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]),
    }

    test_data = Data(args.SET_TYPE, img_dir,EXP_folder, transform=data_transforms['test'])
    testloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


    # optimizer_selection = torch.optim.AdamW(selection.parameters(), lr=args.lr, weight_decay=0.01)
    # selection, optimizer_selection = amp.initialize(selection.cuda(), optimizer_selection, opt_level="O1")

    selection = torch.nn.DataParallel(selection)
    # ----- Test ------
    print('--------Start testing-------')
    selection.eval()
    with torch.no_grad():
        for data in tqdm(testloader):
            art_imgs,port_imgs, labels, demo, lab, IV,k = data
            demo = demo.view(demo.shape[0],1,-1).cuda(non_blocking=True)
            lab = lab.view(lab.shape[0],1,-1).cuda(non_blocking=True).float()
            IV = IV.view(IV.shape[0],1,-1).cuda(non_blocking=True).float()
            art_imgs = art_imgs.cuda(non_blocking=True)
            port_imgs = port_imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            preds = selection(art_imgs,port_imgs, lab, demo,IV)[0]



def main():
    set_random_seed(args.seed)
    # test(args)
if __name__ == '__main__':
    main()
