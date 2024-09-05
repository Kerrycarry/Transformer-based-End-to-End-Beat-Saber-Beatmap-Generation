dora --verbose run  \
    solver=musicgen/musicgen_base_32khz_test \
    model/lm/model_scale=small \
    continue_from=/root/workspace/audiocraft_download/stereo_finetune_musicgen-small.th \
    conditioner=text2music \
    channels=2 interleave_stereo_codebooks.use=True \
    transformer_lm.n_q=8 transformer_lm.card=2048 \
    codebooks_pattern.delay.delays='[0, 0, 1, 1, 2, 2, 3, 3]' \
    --clear