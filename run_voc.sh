
# Step 1. Train SFC for localization maps.

# 1.1 train SFC
CUDA_VISIBLE_DEVICES=0 python train_resnet50_SFC.py
## # 1.2 obtain localization maps
CUDA_VISIBLE_DEVICES=0,1,2,3 python make_cam.py
#### # 1.3 evaluate localization maps
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_cam.py
#
# Step 2. Train IRN for pseudo labels.
# 2.1 generate ir label
CUDA_VISIBLE_DEVICES=0,1,2,3 python cam2ir.py
# # 2.2 train irn
CUDA_VISIBLE_DEVICES=2 python train_irn.py
# 2.3 make pseudo labels
CUDA_VISIBLE_DEVICES=0,1,2,3 python make_seg_labels.py
CUDA_VISIBLE_DEVICES=0 python eval_sem_seg.py

