PRE_SEQ_LEN=1400

CUDA_VISIBLE_DEVICES=0 python3 web_demo.py \
    --model_name_or_path ../THUDM/chatglm2-6b \
    --ptuning_checkpoint output/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-2e-2/checkpoint-3000 \
    --pre_seq_len $PRE_SEQ_LEN

