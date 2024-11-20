nohup ./beatmapgen.sh -3 000 transformer_lm.blockwise_attention_kwargs.local_self_attention=true >> log/beatmapgen_log.txt 2>&1 &
PID=$!
wait $PID

nohup ./beatmapgen.sh -3 001 transformer_lm.blockwise_attention_kwargs.local_self_attention=true --clear >> log/beatmapgen_log.txt 2>&1 &
PID=$!
wait $PID


# mkdir -p log && nohup ./beatmapgen_training.sh > log/beatmapgen_log.txt 2>&1 &