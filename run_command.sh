python run_main.py --description ' Motor imagery EEG signals from the BCIC-IV-2b dataset, with two classes (left-hand and right-hand), represented as discrete token sequences.' \
    --log_dir './logs/' --seed 2026 --model_comment 'EEG Classfication' --model 'EEGLLM' --strategy 'auto' \
    --train_epochs 40 --batch_size 64 --learning_rate 0.0005 --devices 0,1,2 \
    --load_path './DARE-EEG_base.ckpt' --codebook_load_path '../codebook/saved_codebook/' \
    --data 'bcic2a' --data_path '~/data/downtasks/BCIC2020_T3/preprocessed_BCIC2020_T3' --num_subjects 8 \
    --seq_len 512 --enc_in 22 --num_class 4 --patch_size 64 --hidden_dim 256 --depth 8 --num_heads 4 --embed_num 4 --mlp_ratio 4.0 \
    --dropout 0.3 \
    --num_codes 512 --code_dim 256 --beta 0.25 \
    --llm_model GPT2 --llm_dim 768 --llm_layers 6 --n_heads 8 --d_ff 96 \
    --use_lora \
    # --is_binary \
    # if timellm for seedv, set lr to 0.005, epoch to 100
    # GPT2 layers: (12, if too deep set to 6) QWEN3(28, 10)
    # mmwm: Working memory EEG segments for binary cognitive workload classification (low workload and high workload), represented as discrete token sequences.
    # mmwm: ~/data/multi_model/EEG/Sub_S1_single
    # bcic: Motor imagery EEG signals from the BCIC-IV-2b dataset, with two classes (left-hand and right-hand), represented as discrete token sequences.
    # ~/data/downtasks/BCIC_2b_0_38HZ
    # bcic2a: Motor imagery EEG signals from the BCIC-IV-2a dataset, with four classes (left-hand, right-hand, foot, and tongue).
    # seediv:~/data/downtasks/SEED4/processed_data
    # Emotion EEG signals from the SEED-IV dataset, with four classes (happy, sad, fear, and neutral), represented as discrete token sequences.
    # bcic2020_T3
    # Imagined speech EEG segments for five-class classification (hello, help me, stop, thank you, and yes), represented as discrete token sequences.
    # ~/data/downtasks/BCIC2020_T3/preprocessed_BCIC2020_T3