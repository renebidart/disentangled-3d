CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 32 --epochs 400  --save_path ./results/test2_90_aug_opt/avae_modelnet_chair_90aug_all90_l8 --category chair --data_augmentation random_90_rot --opt_method all_90_rot  --val_augmentation random_90_rot --latent_size 8

CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 32 --epochs 400  --save_path ./results/test2_90_aug_opt/avae_modelnet_sofa_90aug_all90_l8 --category sofa --data_augmentation random_90_rot --opt_method all_90_rot  --val_augmentation random_90_rot --latent_size 8

CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 32 --epochs 400  --save_path ./results/test2_90_aug_opt/avae_modelnet_monitor_90aug_all90_l8 --category monitor --data_augmentation random_90_rot --opt_method all_90_rot  --val_augmentation random_90_rot --latent_size 8

CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 32 --epochs 400  --save_path ./results/test2_90_aug_opt/avae_modelnet_toilet_90aug_all90_l8 --category toilet --data_augmentation random_90_rot --opt_method all_90_rot  --val_augmentation random_90_rot --latent_size 8


