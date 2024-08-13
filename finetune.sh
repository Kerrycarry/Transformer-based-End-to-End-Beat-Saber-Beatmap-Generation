dora --verbose run  \
    solver=musicgen/musicgen_base_32khz_test \
    model/lm/model_scale=small \
    continue_from=//pretrained/facebook/musicgen-small \
    conditioner=text2music \
    --clear