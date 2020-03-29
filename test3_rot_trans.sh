CUDA_VISIBLE_DEVICES=1 python train.py --bs 14 --lr 5e-4 --epochs 200  --save_path ./results/test3_5e4/randrottrans_sofa_l16_noopt_r2 --latent_size 16 --category sofa --data_augmentation rand_rot_trans --val_augmentation rand_rot_trans --opt_method none

CUDA_VISIBLE_DEVICES=1 python train.py --bs 14 --lr 5e-4 --epochs 200  --save_path ./results/test3_5e4/randrottrans_sofa_l22_noopt_r2 --latent_size 22 --category sofa --data_augmentation rand_rot_trans --val_augmentation rand_rot_trans --opt_method none

CUDA_VISIBLE_DEVICES=1 python train.py --bs 14 --epochs 200  --save_path ./results/test3_5e4/randrottrans_sofa_l16_opt --latent_size 16 --category sofa --data_augmentation rand_rot_trans --val_augmentation rand_rot_trans --opt_method rand_sgd_rot_trans


CUDA_VISIBLE_DEVICES=1 python train.py --bs 14 --lr 5e-4 --epochs 200  --save_path ./results/test3_5e4/randrottrans_dresser_l16_noopt --latent_size 16 --category dresser --data_augmentation rand_rot_trans --val_augmentation rand_rot_trans --opt_method none

CUDA_VISIBLE_DEVICES=1 python train.py --bs 14 --lr 5e-4 --epochs 200  --save_path ./results/test3_5e4/randrottrans_dresser_l22_noopt --latent_size 22 --category dresser --data_augmentation rand_rot_trans --val_augmentation rand_rot_trans --opt_method none

CUDA_VISIBLE_DEVICES=1 python train.py --bs 14 --lr 5e-4 --epochs 200  --save_path ./results/test3_5e4/randrottrans_dresser_l16_opt --latent_size 16 --category dresser --data_augmentation rand_rot_trans --val_augmentation rand_rot_trans --opt_method rand_sgd_rot_trans

