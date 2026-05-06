export CUDA_VISIBLE_DEVICES=4
python tune_csv.py \
    --experiment brfss_diabetes \
    --model ft_transformer \
    --num_samples 16 \
    --batch_size 4096 \
    --max_epochs 20 \
    --gpus_per_trial 0.5 