MODEL_PATH=/home/dcx/sc/jittor/models/model_0513_2025/iter_mask/sbd_plainvit_base448/002/checkpoints/last_checkpoint.pth
# MODEL_PATH=./weights/pretrained/sbd_vit_base.pth

CUDA_VISIBLE_DEVICES=5 python scripts/evaluate_model.py NoBRS \
--gpu=0 \
--checkpoint=${MODEL_PATH} \
--eval-mode=cvpr \
--print-ious \
--datasets Berkeley \
--min-n-clicks 5