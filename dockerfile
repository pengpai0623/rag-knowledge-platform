# 使用官方 Python 3.10 精简版作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置时区为上海
ENV TZ=Asia/Shanghai

# 安装系统级依赖（包括编译工具，防止某些包没有预编译轮子，顺便把你之前遇到的 swig 也装好）
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 先只复制 requirements.txt，利用 Docker 缓存机制，加快后续构建速度
COPY requirements.txt .

# 安装 Python 依赖（使用清华源加速）
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 将当前目录下的所有代码复制到容器的 /app 目录下
COPY . .

# 暴露 8000 端口
EXPOSE 8000

# 启动 FastAPI 的命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
