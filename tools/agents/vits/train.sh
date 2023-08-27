python scripts/short_audio_transcribe.py --languages "CJE" --whisper_size large


python preprocess_v2.py --languages "CJE"

python preprocess_v2.py --add_auxiliary_data True --languages "CJE"

python finetune_speaker_v2.py -m ./OUTPUT_MODEL --max_epochs "100" --drop_speaker_embed True

python finetune_speaker_v2.py -m ./OUTPUT_MODEL --max_epochs "100" --drop_speaker_embed True --train_with_pretrained_model False

python finetune_speaker_v2.py -m ./OUTPUT_MODEL --max_epochs "100" --drop_speaker_embed False --cont True


tensorboard --logdir=./OUTPUT_MODEL


python VC_inference.py --model_dir ./OUTPUT_MODEL/G_latest.pth