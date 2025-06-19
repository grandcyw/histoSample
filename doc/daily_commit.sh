#!/bin/bash

# 脚本名称：daily_commit.sh
# 功能：自动提交当前更改，Commit 信息为当日日期

# 获取当前日期（格式：YYYY-MM-DD）
COMMIT_DATE=$(date +"%Y-%m-%d")

# Git 操作
git add .
git commit -m "Update ${COMMIT_DATE}: Daily code updates"
git push git@github.com:grandcyw/histoSample.git dev

echo "Daily commit pushed on ${COMMIT_DATE}"