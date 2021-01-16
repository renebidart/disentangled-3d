# EQUIVALENT TO test3 rot trans
# CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --lr 1e-3 --epochs 40 --save_path ./results/mnist/test3/randrottrans_l6_noopt --latent_size 6 --data_augmentation rand_rot_trans --val_augmentation rand_rot_trans --opt_method none

CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --lr 1e-3 --epochs 40 --save_path ./results/mnist/test3/randrottrans_l9_noopt --latent_size 9 --data_augmentation rand_rot_trans --val_augmentation rand_rot_trans --opt_method none

# CUDA_VISIBLE_DEVICES=0 python train_mnist.py --bs 128 --lr 1e-3 --epochs 40 --save_path ./results/mnist/test3/randrottrans_l6_opt --latent_size 6 --data_augmentation rand_rot_trans --val_augmentation rand_rot_trans --opt_method rand_sgd_rot_trans