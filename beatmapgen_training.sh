nohup ./beatmapgen.sh -3 010 transformer_lm.blockwise_attention_kwargs.block_cross_attention=false transformer_lm.blockwise_attention_kwargs.local_cross_attention=true transformer_lm.blockwise_attention_kwargs.singlehead_cross_attention=false >> log/beatmapgen_log.txt 2>&1 &
PID=$!
wait $PID

nohup ./beatmapgen.sh -3 000 transformer_lm.blockwise_attention_kwargs.block_cross_attention=false transformer_lm.blockwise_attention_kwargs.local_cross_attention=false transformer_lm.blockwise_attention_kwargs.singlehead_cross_attention=false >> log/beatmapgen_log.txt 2>&1 &
PID=$!
wait $PID

nohup ./beatmapgen.sh -3 001 transformer_lm.blockwise_attention_kwargs.block_cross_attention=false transformer_lm.blockwise_attention_kwargs.local_cross_attention=false transformer_lm.blockwise_attention_kwargs.singlehead_cross_attention=true >> log/beatmapgen_log.txt 2>&1 &
PID=$!
wait $PID

nohup ./beatmapgen.sh -3 101 transformer_lm.blockwise_attention_kwargs.block_cross_attention=true transformer_lm.blockwise_attention_kwargs.local_cross_attention=false transformer_lm.blockwise_attention_kwargs.singlehead_cross_attention=true >> log/beatmapgen_log.txt 2>&1 &
PID=$!
wait $PID

nohup ./beatmapgen.sh -3 111 transformer_lm.blockwise_attention_kwargs.block_cross_attention=true transformer_lm.blockwise_attention_kwargs.local_cross_attention=true transformer_lm.blockwise_attention_kwargs.singlehead_cross_attention=true >> log/beatmapgen_log.txt 2>&1 &
PID=$!
wait $PID


# mkdir -p log && nohup ./beatmapgen_training.sh > log/beatmapgen_log.txt 2>&1 &