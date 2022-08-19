run Oriented R-CNN:

CUDA_VISIBLE_DEVICES=2 python tools/run_net.py --config-file projects/oriented_rcnn/configs/oriented_rcnn_r101_fpn_1x_dota_ms_with_flip_rotateaug_balance_cate.py

Run Tensorboard on Server-side, monitor on local:

server: tensorboard --logdir=work_dirs

local: ssh -L 16006:127.0.0.1:6006 lyx@localhost -p 9009

local browser: http://127.0.0.1:16006/
