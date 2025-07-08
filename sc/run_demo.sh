# MODEL_PATH=./weights/pretrained/cocolvis_vit_base.pth
MODEL_PATH=/home/dcx/sc/jittor/models/model_0513_2025/iter_mask/sbd_plainvit_base448/002/checkpoints/last_checkpoint.pth
python3 demo.py \
--checkpoint=${MODEL_PATH} \
--gpu 0