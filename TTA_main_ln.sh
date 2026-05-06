export CUDA_VISIBLE_DEVICES=1
for exp in acsfoodstamps acsunemployment assistments brfss_diabetes nhanes_lead physionet; do
    python TTA_main_csv_ln.py --experiment $exp --model ft_transformer --n_epochs 3 --batch_size 1024
done
