#!/bin/bash

ARGS="--rdzv-backend=c10d \
    --rdzv-endpoint=$rdzv_endpoint \
    --nnodes=$nnodes \
    --max-restarts=$max_restarts \
    --nproc-per-node=$num_gpus \
    main.py \
    --do_eval $do_eval \
    --do_train $do_train \
    --do_predict $do_predict \
    --preprocessing_num_workers 10 \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path ../THUDM/chatglm2-6b \
    --output_dir output/$out_prefix-chatglm2-6b-pt-$pre_seq_len-$lr \
    --overwrite_output_dir \
    --max_source_length $pre_seq_len \
    --max_target_length $max_target_length \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --predict_with_generate \
    --max_steps $max_steps \
    --logging_steps 10 \
    --save_steps $save_steps \
    --learning_rate $lr \
    --pre_seq_len $pre_seq_len"
    # --quantization_bit 4

if [ -n "$ptuning_checkpoint" ]; then
    ARGS=$ARGS"--ptuning_checkpoint $ptuning_checkpoint"
fi

if [ "$do_train" = "True" ]; then
    ARGS=$ARGS" --train_file $train_dir/train.json"
fi

if [ "$do_eval" = "True" ]; then
    ARGS=$ARGS" --validation_file $train_dir/dev.json"
fi

if [ "$do_predict" = "True" ]; then
    ARGS=$ARGS" --test_file $train_dir/test.json"
fi

echo $ARGS

torchrun $ARGS