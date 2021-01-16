CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_sofa_aug_noopt_l64 --model_type avae3d --category sofa --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 64

CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_sofa_aug_noopt_l32 --model_type avae3d --category sofa --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 32 

CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_sofa_aug_noopt_l16 --model_type avae3d --category sofa --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 16 

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_sofa_aug_noopt_l8 --model_type avae3d --category sofa --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 8 

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_sofa_aug_noopt_l4 --model_type avae3d --category sofa --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 4



CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_sofa_none_noopt_l64 --model_type avae3d --category sofa --data_augmentation none --val_augmentation none --opt_method none  --latent_size 64 

CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_sofa_none_noopt_l32 --model_type avae3d --category sofa --data_augmentation none --val_augmentation none --opt_method none  --latent_size 32 

CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_sofa_none_noopt_l16 --model_type avae3d --category sofa --data_augmentation none --val_augmentation none --opt_method none  --latent_size 16 

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_sofa_none_noopt_l8 --model_type avae3d --category sofa --data_augmentation none --val_augmentation none --opt_method none  --latent_size 8 

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_sofa_none_noopt_l4 --model_type avae3d --category sofa --data_augmentation none --val_augmentation none --opt_method none  --latent_size 4 


# CHAIR
CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_chair_aug_noopt_l64 --model_type avae3d --category chair --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 64 

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_chair_aug_noopt_l32 --model_type avae3d --category chair --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 32 

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_chair_aug_noopt_l16 --model_type avae3d --category chair --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 16 

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_chair_aug_noopt_l8 --model_type avae3d --category chair --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 8 

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_chair_aug_noopt_l4 --model_type avae3d --category chair --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 4 


CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_chair_none_noopt_l64 --model_type avae3d --category chair --data_augmentation none --val_augmentation none --opt_method none  --latent_size 64 
# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_chair_none_noopt_l32 --model_type avae3d --category chair --data_augmentation none --val_augmentation none --opt_method none  --latent_size 32 

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_chair_none_noopt_l16 --model_type avae3d --category chair --data_augmentation none --val_augmentation none --opt_method none  --latent_size 16 

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_chair_none_noopt_l8 --model_type avae3d --category chair --data_augmentation none --val_augmentation none --opt_method none  --latent_size 8 

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_chair_none_noopt_l4 --model_type avae3d --category chair --data_augmentation none --val_augmentation none --opt_method none  --latent_size 4 


### BED
CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_bed_aug_noopt_l64 --model_type avae3d --category bed --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 64 

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_bed_aug_noopt_l32 --model_type avae3d --category bed --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 32 

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_bed_aug_noopt_l16 --model_type avae3d --category bed --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 16 

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_bed_aug_noopt_l8 --model_type avae3d --category bed --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 8 

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_bed_aug_noopt_l4 --model_type avae3d --category bed --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 4 

CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_bed_none_noopt_l64 --model_type avae3d --category bed --data_augmentation none --val_augmentation none --opt_method none  --latent_size 64 

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_bed_none_noopt_l32 --model_type avae3d --category bed --data_augmentation none --val_augmentation none --opt_method none  --latent_size 32 

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_bed_none_noopt_l16 --model_type avae3d --category bed --data_augmentation none --val_augmentation none --opt_method none  --latent_size 16 

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_bed_none_noopt_l8 --model_type avae3d --category bed --data_augmentation none --val_augmentation none --opt_method none  --latent_size 8 

# CUDA_VISIBLE_DEVICES=0 python train.py --lr 3e-3 --bs 64 --epochs 200  --save_path ./results/test1_aug_noopt/avae_modelnet_bed_none_noopt_l4 --model_type avae3d --category bed --data_augmentation none --val_augmentation none --opt_method none  --latent_size 4 

# ??? Monitor