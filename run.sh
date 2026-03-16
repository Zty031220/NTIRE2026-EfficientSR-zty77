# --- Evaluation on LSDIR_DIV2K_valid datasets for One Method: ---
CUDA_VISIBLE_DEVICES=0 python test_demo.py \
    --data_dir ../ \
    --save_dir ../results \
    --model_id 05


# --- When only LSDIR_DIV2K_test datasets are included (For Organizer) ---
#CUDA_VISIBLE_DEVICES=0 python test_demo.py \
#     --data_dir ../ \
#     --save_dir ../results \
#     --include_test \
#     --model_id 05
