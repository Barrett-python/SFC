
# Step 1. Train SIPE for localization maps.

# 1.1 train sipe
CUDA_VISIBLE_DEVICES=2,3 python train_resnet50_SIPE.py  --dataset coco  --session_name exp_coco
# 1.2 obtain localization maps
# CUDA_VISIBLE_DEVICES=2,3 python make_cam.py --dataset coco  --session_name exp_coco
# 1.3 evaluate localization maps
# CUDA_VISIBLE_DEVICES=2,3 python eval_cam.py --dataset coco  --session_name exp_coco


# Step 2. Refinement for pseudo labels.
# Note: for MS COCO dataset, we replace IRN with denseCRF due to the large computation cost.
# CUDA_VISIBLE_DEVICES=2,3 python cam2ir.py --dataset coco  --session_name exp_coco

