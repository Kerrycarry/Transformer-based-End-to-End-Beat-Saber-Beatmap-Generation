#!/bin/bash

# 启动 parser API
start_parser_api() {
    echo "Starting Deno API..."
    nohup deno run --allow-net --allow-read --allow-write audiocraft/data/beatmap_parser.ts > beatmapgen_log_parser_api.txt 2>&1 &
    API_PID=$!
    sleep 5
    echo "Deno API PID: $API_PID"
}

# 流程 1：Dora 主流程 1
run_dora_process_1() {
    echo "Running beatmapgen pre compute token ..."
    nohup dora run \
        solver=beatmapgen/beatmapgen_base_32khz \
        model/lm/model_scale=xsmall \
        conditioner=text2music \
        cache.path=/mnt/workspace/cache/bs_rank \
        cache.write=True \
        --clear > beatmapgen_log_dora.txt 2>&1 &
    DORA_PID=$!
    wait $DORA_PID
}

# 流程 2：Dora 主流程 2
run_dora_process_2() {
    echo "Running beatmapgen with precomputerd token..."
    nohup dora run \
        solver=beatmapgen/beatmapgen_base_32khz \
        model/lm/model_scale=small \
        conditioner=text2music \
        cache.path=/mnt/workspace/cache/bs_rank \
        continue_from=/root/workspace/audiocraft_download/beatmapgen_finetune_musicgen-small.th \
        --clear > beatmapgen_log_dora.txt 2>&1 &
    DORA_PID=$!
    wait $DORA_PID
}

# manifest pipeline
run_python_process() {
    echo "Running manifest pipeline..."
    nohup python -m audiocraft.data.audio_dataset_beatmap \
        dataset/CustomLevels egs/bs_rank/data.jsonl 0.125 --write_parse_switch > beatmapgen_log_dora.txt 2>&1 &
    PYTHON_PID=$!
    wait $PYTHON_PID
}

# 终止 Deno API
stop_parser_api() {
    echo "Stopping Deno API..."
    kill -9 $API_PID
    echo "Deno API stopped."
}

kill_process() {
    PROCESS_NAME=$1
    PID=$(ps aux | grep "$PROCESS_NAME" | grep -v grep | awk '{print $2}')
    if [ -z "$PID" ]; then
        echo "$PROCESS_NAME is not running."
    else
        echo "Killing $PROCESS_NAME process (PID: $PID)..."
        kill $PID
        if [ $? -eq 0 ]; then
            echo "$PROCESS_NAME process killed successfully."
        else
            echo "Failed to kill $PROCESS_NAME process."
        fi
    fi
}

# 根据参数运行不同的流程
if [[ "$1" == "-1" ]]; then
    start_parser_api
    run_dora_process_1
    stop_parser_api
elif [[ "$1" == "-2" ]]; then
    start_parser_api
    run_dora_process_2
    stop_parser_api
elif [[ "$1" == "-3" ]]; then
    start_parser_api
    run_python_process
    stop_parser_api
elif [[ "$1" == "-4" ]]; then
    for PROCESS in deno dora python
    do
        kill_process "$PROCESS"
    done
else
    echo "Invalid option. Use -1 or -2 to select the process."
fi

# usage 
# nohup ./beatmapgen.sh -1 > beatmapgen_log.txt 2>&1 &
# nohup ./beatmapgen.sh -2 > beatmapgen_log.txt 2>&1 &
# nohup ./beatmapgen.sh -3 > beatmapgen_log.txt 2>&1 &
# ./beatmapgen.sh -4

