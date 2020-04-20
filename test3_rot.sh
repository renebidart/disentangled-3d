# CUDA_VISIBLE_DEVICES=0 python train.py --bs 14 --lr 5e-4 --epochs 200  --save_path ./results/test3_5e4/randrot_sofa_l16_noopt --latent_size 16 --category sofa --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none

# CUDA_VISIBLE_DEVICES=0 python train.py --bs 14 --lr 5e-4 --epochs 200  --save_path ./results/test3_5e4/randrot_sofa_l22_noopt --latent_size 22 --category sofa --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none

# CUDA_VISIBLE_DEVICES=0 python train.py --bs 14 --lr 5e-4 --epochs 200  --save_path ./results/test3_5e4/randrot_sofa_l16_opt --latent_size 16 --category sofa --data_augmentation rand_rot --val_augmentation rand_rot --opt_method rand_sgd_rot


CUDA_VISIBLE_DEVICES=0 python train.py --bs 14 --lr 5e-4 --epochs 100  --save_path ./results/test3_5e4/randrot_chair_l16_noopt_r2 --latent_size 16 --category chair --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none

CUDA_VISIBLE_DEVICES=0 python train.py --bs 14 --lr 5e-4 --epochs 100  --save_path ./results/test3_5e4/randrot_chair_l22_noopt_r2 --latent_size 22 --category chair --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none

CUDA_VISIBLE_DEVICES=0 python train.py --bs 14 --lr 5e-4 --epochs 100  --save_path ./results/test3_5e4/randrot_chair_l16_opt --latent_size 16 --category chair --data_augmentation rand_rot --val_augmentation rand_rot --opt_method rand_sgd_rot