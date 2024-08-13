dora --verbose run  \
    solver=musicgen/musicgen_base_32khz_test \
    model/lm/model_scale=xsmall \
    conditioner=none \
    cache.path=/tmp/cache/interleave_stereo_nv_32k \
    cache.write=True \
    --clear