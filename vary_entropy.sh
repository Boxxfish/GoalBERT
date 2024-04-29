# python train.py --v_net_finetune experiments/exp-v9gh4m3p/checkpoints/v_net-20.safetensors --num_masks 4 --iterations 500 --max_hops 2 --entropy_coeff 0.01
# python train.py --v_net_finetune experiments/exp-v9gh4m3p/checkpoints/v_net-20.safetensors --num_masks 4 --iterations 500 --max_hops 2 --entropy_coeff 0.03
python train.py --v_net_finetune experiments/exp-v9gh4m3p/checkpoints/v_net-20.safetensors --num_masks 4 --iterations 500 --max_hops 2 --entropy_coeff 0.3
