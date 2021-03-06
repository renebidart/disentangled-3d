# CUDA_VISIBLE_DEVICES=1 python train.py --lr 3e-3 --bs 64 --epochs 400  --save_path ./results/test1_aug_noopt/avae_modelnet_1class_90aug_noopt_l32 --model_type avae3d --n_classes 1 --data_augmentation random_90_rot --val_augmentation random_90_rot --opt_method none  --latent_size 32 
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 400  --save_path ./results/test1_aug_noopt/avae_modelnet_10class_90aug_noopt_l32 --model_type avae3d --n_classes 10 --data_augmentation random_90_rot --val_augmentation random_90_rot --opt_method none  --latent_size 32

# CUDA_VISIBLE_DEVICES=1 python train.py --lr 3e-3 --bs 64 --epochs 400  --save_path ./results/test1_aug_noopt/avae_modelnet_1class_90aug_noopt_l16 --model_type avae3d --n_classes 1 --data_augmentation random_90_rot --val_augmentation random_90_rot --opt_method none  --latent_size 16 
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 400  --save_path ./results/test1_aug_noopt/avae_modelnet_10class_90aug_noopt_l16 --model_type avae3d --n_classes 10 --data_augmentation random_90_rot --val_augmentation random_90_rot --opt_method none  --latent_size 16

# CUDA_VISIBLE_DEVICES=1 python train.py --lr 3e-3 --bs 64 --epochs 400  --save_path ./results/test1_aug_noopt/avae_modelnet_1class_90aug_noopt_l8 --model_type avae3d --n_classes 1 --data_augmentation random_90_rot --val_augmentation random_90_rot --opt_method none  --latent_size 8 
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 400  --save_path ./results/test1_aug_noopt/avae_modelnet_10class_90aug_noopt_l8 --model_type avae3d --n_classes 10 --data_augmentation random_90_rot --val_augmentation random_90_rot --opt_method none  --latent_size 8

# CUDA_VISIBLE_DEVICES=1 python train.py --lr 3e-3 --bs 64 --epochs 400  --save_path ./results/test1_aug_noopt/avae_modelnet_1class_90aug_noopt_l4 --model_type avae3d --n_classes 1 --data_augmentation random_90_rot  --val_augmentation random_90_rot --opt_method none  --latent_size 4 
CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 400  --save_path ./results/test1_aug_noopt/avae_modelnet_10class_90aug_noopt_l4 --model_type avae3d --n_classes 10 --data_augmentation random_90_rot --val_augmentation random_90_rot --opt_method none  --latent_size 4





# CUDA_VISIBLE_DEVICES=1 python train.py --lr 3e-3 --bs 64 --epochs 400  --save_path ./results/test1_aug_noopt/avae_modelnet_1class_noaug_noopt_l32 --model_type avae3d --n_classes 1 --data_augmentation none --opt_method none  --latent_size 32 
CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 400  --save_path ./results/test1_aug_noopt/avae_modelnet_10class_noaug_noopt_l32 --model_type avae3d --n_classes 10 --data_augmentation none --opt_method none  --latent_size 32

# CUDA_VISIBLE_DEVICES=1 python train.py --lr 3e-3 --bs 64 --epochs 400  --save_path ./results/test1_aug_noopt/avae_modelnet_1class_noaug_noopt_l16 --model_type avae3d --n_classes 1 --data_augmentation none --opt_method none  --latent_size 16 
CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 400  --save_path ./results/test1_aug_noopt/avae_modelnet_10class_noaug_noopt_l16 --model_type avae3d --n_classes 10 --data_augmentation none --opt_method none  --latent_size 16

# CUDA_VISIBLE_DEVICES=1 python train.py --lr 3e-3 --bs 64 --epochs 400  --save_path ./results/test1_aug_noopt/avae_modelnet_1class_noaug_noopt_l8 --model_type avae3d --n_classes 1 --data_augmentation none --opt_method none  --latent_size 8 
CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 400  --save_path ./results/test1_aug_noopt/avae_modelnet_10class_noaug_noopt_l8 --model_type avae3d --n_classes 10 --data_augmentation none --opt_method none  --latent_size 8

# CUDA_VISIBLE_DEVICES=1 python train.py --lr 3e-3 --bs 64 --epochs 400  --save_path ./results/test1_aug_noopt/avae_modelnet_1class_noaug_noopt_l4 --model_type avae3d --n_classes 1 --data_augmentation none --opt_method none  --latent_size 4 
CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 400  --save_path ./results/test1_aug_noopt/avae_modelnet_10class_noaug_noopt_l4 --model_type avae3d --n_classes 10 --data_augmentation none --opt_method none  --latent_size 4
