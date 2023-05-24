from builtins import bool
import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import importlib
import os
import imageio
import argparse
from data import data_voc, data_coco
from tool import torchutils, pyutils
from tool import torchutils, imutils
import torch
cudnn.enabled = True

def overlap(img, hm):
    hm = plt.cm.jet(hm)[:, :, :3]
    hm = np.array(Image.fromarray((hm*255).astype(np.uint8), 'RGB').resize((img.shape[1], img.shape[0]), Image.BICUBIC)).astype(np.float)*2
    if hm.shape == np.array(img).astype(np.float).shape:
        out = (hm + np.array(img).astype(np.float)) / 3
        out = (out / np.max(out) * 255).astype(np.uint8)
    else:
        print(hm.shape)
        print(np.array(img).shape)
    return out

def draw_heatmap(norm_cam, gt_label, orig_img, save_path, img_name):
    gt_cat = np.where(gt_label==1)[0]
    for _, gt in enumerate(gt_cat):
        heatmap = overlap(orig_img, norm_cam[gt])
        cam_viz_path = os.path.join(save_path, img_name + '_{}.png'.format(gt))
        imageio.imsave(cam_viz_path, heatmap)


def _work(process_id, model, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0] # 2007_000032
            label = pack['label'][0] # torch.Size([21])
            size = pack['size'] # tensor([281]) 1:tensor([500])
            label = F.pad(label, (1, 0), 'constant', 1.0) # torch.Size([21])

            outputs = [model(img[0].cuda(non_blocking=True), label.cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)) for img in pack['img']]
            
            # multi-scale fusion
            IS_CAM_list = [output[1].cpu() for output in outputs] # 4 * torch.Size([21, 18, 32])
            IS_CAM_list = [F.interpolate(torch.unsqueeze(o, 1), size, mode='bilinear', align_corners=False) for o in IS_CAM_list] # torch.Size([21, 1, 281, 500])
            IS_CAM = torch.sum(torch.stack(IS_CAM_list, 0), 0)[:,0] # torch.Size([21, 281, 500])
            
            IS_CAM /= F.adaptive_max_pool2d(IS_CAM, (1, 1)) + 1e-5 # torch.Size([21, 281, 500])
            
            IS_CAM = IS_CAM + np.pad(IS_CAM[1:, ...], ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.4)
            IS_CAM = IS_CAM.cpu().numpy()
            prototye = IS_CAM
            # save IS_CAM
            valid_cat = torch.nonzero(label)[:, 0].cpu().numpy()
            IS_CAM = IS_CAM[valid_cat]

            IS_CAM_list1 = [output[0].cpu() for output in outputs] # 4 * torch.Size([21, 18, 32])
            IS_CAM_list1 = [F.interpolate(torch.unsqueeze(o, 1), size, mode='bilinear', align_corners=False) for o in IS_CAM_list1] # torch.Size([21, 1, 281, 500])
            IS_CAM1 = torch.sum(torch.stack(IS_CAM_list1, 0), 0)[:,0] # torch.Size([21, 281, 500])
            IS_CAM1 /= F.adaptive_max_pool2d(IS_CAM1, (1, 1)) + 1e-5 # torch.Size([21, 281, 500])
            IS_CAM1 = IS_CAM1 + np.pad(IS_CAM1[1:, ...], ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.4)
            IS_CAM1 = IS_CAM1.cpu().numpy()
            cam_classifier = IS_CAM1
            # save IS_CAM
            valid_cat = torch.nonzero(label)[:, 0].cpu().numpy()
            IS_CAM1 = IS_CAM1[valid_cat]
           
#            final_cam = prototye*0.3 + cam_classifier
#            if args.visualize:
#                orig_img = np.array(Image.open(pack['img_path'][0]).convert('RGB'))
#                draw_heatmap(cam_classifier.copy(), label, orig_img, os.path.join(args.session_name, 'visual_classifier'), img_name)
#                draw_heatmap(prototye.copy(), label, orig_img, os.path.join(args.session_name, 'visual_prototye'), img_name)
#                draw_heatmap(final_cam.copy(), label, orig_img, os.path.join(args.session_name, 'final_cam'), img_name)

            np.save(os.path.join(args.session_name, 'npy', img_name + '.npy'),  {"keys": valid_cat, "IS_CAM": IS_CAM, "IS_CAM1": IS_CAM1})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="network.resnet50_SIPE_Best", type=str)
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--session_name", default="exp", type=str)
    parser.add_argument("--ckpt", default="best.pth", type=str)
    parser.add_argument("--visualize", default=True, type=bool)
    parser.add_argument("--dataset", default="voc", type=str)

    args = parser.parse_args()

    os.makedirs(os.path.join(args.session_name, 'npy'), exist_ok=True)
    os.makedirs(os.path.join(args.session_name, 'visual_classifier'), exist_ok=True)
    os.makedirs(os.path.join(args.session_name, 'visual_prototye'), exist_ok=True)

    pyutils.Logger(os.path.join(args.session_name, 'infer.log'))
    print(vars(args))

    assert args.dataset in ['voc', 'coco'], 'Dataset must be voc or coco in this project.'

    if args.dataset == 'voc':
        dataset_root = '/data/tfl/VOCdevkit/VOC2012'
        model = getattr(importlib.import_module(args.network), 'CAM')(num_cls=21)
        dataset = data_voc.VOC12ClsDatasetMSF('data/trainaug_' + args.dataset + '.txt', voc12_root=dataset_root, scales=(1.0, 0.5, 1.5, 2.0))

    elif args.dataset == 'coco':
        dataset_root = "/data/tfl/COCO/"
        model = getattr(importlib.import_module(args.network), 'CAM')(num_cls=81)
        dataset = data_coco.COCOClsDatasetMSF('data/train_' + args.dataset + '.txt', coco_root=dataset_root, scales=(1.0, 0.5, 1.5, 2.0))

    checkpoint = torch.load(args.session_name + '/ckpt/' + args.ckpt)
    model.load_state_dict(checkpoint['net'], strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='') 
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()