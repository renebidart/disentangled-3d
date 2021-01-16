# THESE TWO ARE FORGOTTEN BASELINES FOR ROTT AUG TESTS
CUDA_VISIBLE_DEVICES=1 python train.py --bs 14 --lr 5e-4 --epochs 100  --save_path ./results/test3_5e4/randrot_sofa_l19_noopt --latent_size 19 --category sofa --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none
CUDA_VISIBLE_DEVICES=1 python train.py --bs 14 --lr 5e-4 --epochs 100  --save_path ./results/test3_5e4/randrot_chair_l19_noopt --latent_size 19 --category chair --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none


CUDA_VISIBLE_DEVICES=1 python train.py --bs 14 --lr 8e-4 --epochs 50  --save_path ./results/test3_1d/randrot1d_sofa_l8_noopt --latent_size 8 --category sofa --data_augmentation rand_rot_1d --val_augmentation rand_rot_1d --opt_method none
CUDA_VISIBLE_DEVICES=1 python train.py --bs 14 --lr 8e-4 --epochs 50  --save_path ./results/test3_1d/randrot1d_sofa_l9_noopt --latent_size 9 --category sofa --data_augmentation rand_rot_1d --val_augmentation rand_rot_1d --opt_method none
CUDA_VISIBLE_DEVICES=1 python train.py --bs 14 --lr 8e-4 --epochs 50  --save_path ./results/test3_1d/randrot1d_sofa_l8_opt --latent_size 8 --category sofa --data_augmentation rand_rot_1d --val_augmentation rand_rot_1d --opt_method rand_sgd_1d


# CUDA_VISIBLE_DEVICES=1 python train.py --bs 14 --lr 5e-4 --epochs 100  --save_path ./results/test3_5e4/randrot1d_chair_l16_noopt --latent_size 16 --category chair --data_augmentation rand_rot_1d --val_augmentation rand_rot_1d --opt_method none

# CUDA_VISIBLE_DEVICES=1 python train.py --bs 14 --lr 5e-4 --epochs 100  --save_path ./results/test3_5e4/randrot1d_chair_l22_noopt --latent_size 22 --category chair --data_augmentation rand_rot_1d --val_augmentation rand_rot_1d --opt_method none

# CUDA_VISIBLE_DEVICES=1 python train.py --bs 14 --lr 5e-4 --epochs 100  --save_path ./results/test3_5e4/randrot1d_chair_l16_opt --latent_size 16 --category chair --data_augmentation rand_rot_1d --val_augmentation rand_rot_1d --opt_method rand_sgd_rot

# CUDA_VISIBLE_DEVICES=1 python train.py --bs 14 --lr 5e-4 --epochs 100  --save_path ./results/test3_5e4/randrot1d_sofa_l16_opt --latent_size 16 --category sofa --data_augmentation rand_rot_1d --val_augmentation rand_rot_1d --opt_method rand_sgd_1d


CUDA_VISIBLE_DEVICES=1 python train.py --bs 14 --lr 8e-4 --epochs 50  --save_path ./results/test3_1d/randrot1d_chair_l8_noopt --latent_size 8 --category chair --data_augmentation rand_rot_1d --val_augmentation rand_rot_1d --opt_method none
CUDA_VISIBLE_DEVICES=1 python train.py --bs 14 --lr 8e-4 --epochs 50  --save_path ./results/test3_1d/randrot1d_chair_l9_noopt --latent_size 9 --category chair --data_augmentation rand_rot_1d --val_augmentation rand_rot_1d --opt_method none
CUDA_VISIBLE_DEVICES=1 python train.py --bs 14 --lr 8e-4 --epochs 50  --save_path ./results/test3_1d/randrot1d_chair_l8_opt --latent_size 8 --category chair --data_augmentation rand_rot_1d --val_augmentation rand_rot_1d --opt_method rand_sgd_1d