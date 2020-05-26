# CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_none_noopt_l64 --data_augmentation none --val_augmentation none --opt_method none  --latent_size 64

# CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_none_noopt_l32 --data_augmentation none --val_augmentation none --opt_method none  --latent_size 32

# CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_none_noopt_l16 --data_augmentation none --val_augmentation none --opt_method none  --latent_size 16 

# CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_none_noopt_l8 --data_augmentation none --val_augmentation none --opt_method none  --latent_size 8 

# CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_none_noopt_l4 --data_augmentation none --val_augmentation none --opt_method none  --latent_size 4 


# CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_aug_noopt_l64 --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 64

# CUDA_VISIBLE_DEVICES=1 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_aug_noopt_l32 --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 32 

# CUDA_VISIBLE_DEVICES=1 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_aug_noopt_l16 --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 16 

# CUDA_VISIBLE_DEVICES=1 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_aug_noopt_l8 --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 8 

# CUDA_VISIBLE_DEVICES=1 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_aug_noopt_l4 --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 4 

# # rerun
# CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_none_noopt_l64 --data_augmentation none --val_augmentation none --opt_method none  --latent_size 64



# # EQUIVALENT TO test3 rot
# CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --lr 1e-3 --epochs 100  --save_path ./results/mnist/test3/randrot_l16_noopt --latent_size 16 --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none

# CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --lr 1e-3 --epochs 100  --save_path ./results/mnist/test3/randrot_l22_noopt --latent_size 22 --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none

# CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --lr 1e-3 --epochs 100  --save_path ./results/mnist/test3/randrot_l16_opt --latent_size 16 --data_augmentation rand_rot --val_augmentation rand_rot --opt_method rand_sgd_rot

CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --lr 1e-3 --epochs 40  --save_path ./results/mnist/test3/randrot_l6_noopt --latent_size 6 --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none

CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --lr 1e-3 --epochs 40  --save_path ./results/mnist/test3/randrot_l7_noopt --latent_size 7 --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none

CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --lr 1e-3 --epochs 40  --save_path ./results/mnist/test3/randrot_l6_opt --latent_size 6 --data_augmentation rand_rot --val_augmentation rand_rot --opt_method rand_sgd_rot