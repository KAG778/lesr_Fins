#!/bin/bash
# 加载项目环境变量
export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
echo "环境变量已加载"
