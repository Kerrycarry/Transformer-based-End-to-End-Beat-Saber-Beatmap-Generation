dora --verbose run  \
    solver=musicgen/beatmapgen_base_32khz_test \
    model/lm/model_scale=small \
    continue_from=/root/workspace/audiocraft_download/beatmapgen_finetune_musicgen-small.th \
    conditioner=none \
    --clear