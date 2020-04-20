# CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_none_noopt_l64 --data_augmentation none --val_augmentation none --opt_method none  --latent_size 64

CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_none_noopt_l32 --data_augmentation none --val_augmentation none --opt_method none  --latent_size 32 

CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_none_noopt_l16 --data_augmentation none --val_augmentation none --opt_method none  --latent_size 16 

CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_none_noopt_l8 --data_augmentation none --val_augmentation none --opt_method none  --latent_size 8 

CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_none_noopt_l4 --data_augmentation none --val_augmentation none --opt_method none  --latent_size 4 


# CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_aug_noopt_l64 --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 64

CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_aug_noopt_l32 --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 32 

CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_aug_noopt_l16 --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 16 

CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_aug_noopt_l8 --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 8 

CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --epochs 200  --save_path ./results/mnist/test1/avae_aug_noopt_l4 --data_augmentation rand_rot --val_augmentation rand_rot --opt_method none  --latent_size 4 

