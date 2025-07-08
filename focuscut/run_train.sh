# CUDA_VISIBLE_DEVICES=0 python main.py -rf -ls multistep
# CUDA_VISIBLE_DEVICES=7 python main.py -rf -ap "backbone='resnet101'" -ls multistep --val_robot_interval 100
CUDA_VISIBLE_DEVICES=7 python main.py -rf -ap "backbone='resnet101'" -dv GrabCut