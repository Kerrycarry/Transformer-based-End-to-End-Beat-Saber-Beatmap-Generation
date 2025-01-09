#!/bin/bash

# 启动 parser API
start_parser_api() {
    echo "Starting Deno API..."
    nohup deno run --allow-net --allow-read --allow-write audiocraft/data/beatmap_parser.ts > "$LOG_FILE2" 2>&1 &
    API_PID=$!
    sleep 5
    echo "Deno API PID: $API_PID"
}

#
run_dora_process() {
    echo "Running beatmapgen without caching"
    nohup dora run \
        solver=beatmapgen/beatmapgen_base_32khz \
        model/lm/model_scale=small \
        conditioner=none \
        ${MORE_CONFIG[@]} > "$LOG_FILE" 2>&1 &
    DORA_PID=$!
    wait $DORA_PID
}


# manifest pipeline
run_python_process() {
    echo "Running manifest pipeline..."
    nohup python -m audiocraft.data.audio_dataset_beatmap \
        dataset/$SOURCE_DIR egs/$MANIFEST_DIR/data.jsonl 0.125 \
        --write_parse_switch > "$LOG_FILE" 2>&1 &
        # > "$LOG_FILE" 2>&1 &

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

echo "Calling command: $0 $@" 
FIRST_ARG=$1
TAG=$2
if [[ "$FIRST_ARG" == "-2" ]]; then
    SOURCE_DIR=$2
    MANIFEST_DIR=$3
fi
shift  # 移除第一个参数
shift  # 移除第二个参数
MORE_CONFIG=("$@")  # 捕获剩余的参数到数组中

LOG_DIR="log" # 定义日志文件夹
LOG_FILE="${LOG_DIR}/beatmapgen_log_${TAG}.dora"
LOG_FILE2="${LOG_DIR}/beatmapgen_log_${TAG}.api"

# 确保日志文件夹存在
mkdir -p "$LOG_DIR"

# 根据参数运行不同的流程
if [[ "$FIRST_ARG" == "-1" ]]; then
    start_parser_api
    run_dora_process
    stop_parser_api
elif [[ "$FIRST_ARG" == "-2" ]]; then
    TAG="manifest"
    LOG_FILE="${LOG_DIR}/beatmapgen_log_${TAG}.dora"
    LOG_FILE2="${LOG_DIR}/beatmapgen_log_${TAG}.api"
    start_parser_api
    run_python_process
    stop_parser_api
elif [[ "$FIRST_ARG" == "-3" ]]; then
    start_parser_api
elif [[ "$FIRST_ARG" == "-4" ]]; then
    for PROCESS in deno dora
    do
        kill_process "$PROCESS"
    done
else
    echo "Invalid option. Use -1 or -2 to select the process."
fi

# usage 
# 启动主流程
# nohup ./beatmapgen.sh -1 default > log/beatmapgen_log.txt 2>&1 &
# nohup ./beatmapgen.sh -1 default --clear > log/beatmapgen_log.txt 2>&1 &
# 启动manifest制作流程
# nohup ./beatmapgen.sh -2 CustomLevels2 bs_curated > log/beatmapgen_log.txt 2>&1 &
# 启动 api
# ./beatmapgen.sh -3
# kill 掉api和dora
# ./beatmapgen.sh -4


