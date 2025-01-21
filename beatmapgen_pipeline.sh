nohup ./beatmapgen.sh -2 CustomLevels bs_rank config/solver/beatmapgen/beatmapgen_base_32khz.yaml create_manifest >> log/beatmapgen_log.txt 2>&1 &
PID=$!
wait $PID
nohup ./beatmapgen.sh -2 CustomLevels bs_rank config/solver/beatmapgen/beatmapgen_base_32khz.yaml tokenize_beatmap >> log/beatmapgen_log.txt 2>&1 &
PID=$!
wait $PID
nohup ./beatmapgen.sh -2 CustomLevels bs_rank config/solver/beatmapgen/beatmapgen_base_32khz.yaml tokenize_audio >> log/beatmapgen_log.txt 2>&1 &
PID=$!
wait $PID


# mkdir -p log && nohup ./beatmapgen_pipeline.sh > log/beatmapgen_log.txt 2>&1 &