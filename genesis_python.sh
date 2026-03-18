#!/bin/bash

# 1. 注入我们之前测试成功的无头渲染环境变量
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export PYGLET_HEADLESS=1

# 2. 定位你的虚拟环境 Python 绝对路径
# (请确认路径是否正确，这里假设你的环境在 ~/Genesis/.venv 中)
VENV_PYTHON="/jet/home/xxiong1/Genesis/.venv/bin/python"

# 3. 核心魔法：使用 Apptainer 带着 GPU 权限执行这个 Python，并把 VS Code 传来的所有参数 ("$@") 原样转发进去！
apptainer exec --nv $PROJECT/containers/genesis_full.sif $VENV_PYTHON "$@"