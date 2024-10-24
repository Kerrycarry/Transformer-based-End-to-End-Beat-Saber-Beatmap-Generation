#!/bin/bash

# 查找并终止 Deno 进程
DENO_PID=$(ps aux | grep 'deno' | grep -v grep | awk '{print $2}')
if [ -z "$DENO_PID" ]; then
    echo "Deno is not running."
else
    echo "Killing Deno process (PID: $DENO_PID)..."
    kill $DENO_PID
    if [ $? -eq 0 ]; then
        echo "Deno process killed successfully."
    else
        echo "Failed to kill Deno process."
    fi
fi

# 查找并终止 Dora 进程
DORA_PID=$(ps aux | grep 'dora' | grep -v grep | awk '{print $2}')
if [ -z "$DORA_PID" ]; then
    echo "Dora is not running."
else
    echo "Killing Dora process (PID: $DORA_PID)..."
    kill $DORA_PID
    if [ $? -eq 0 ]; then
        echo "Dora process killed successfully."
    else
        echo "Failed to kill Dora process."
    fi
fi
