export CUDA_VISIBLE_DEVICES=0
for exp in assistments; do                                                          
    python TTA_main_csv.py --experiment $exp --model ft_transformer --n_epochs 1 --batch_size 2048                                                     
done  
# acsfoodstamps acsunemployment brfss_diabetes nhanes_lead physionet;