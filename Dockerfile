FROM python:3.12-slim

WORKDIR /app

# 先拷依赖清单再安装：改代码时这一层能命中缓存，不用重装
COPY requirements-serve.txt .

# 国内直连 PyPI 很慢，默认走清华镜像；境外构建可用 --build-arg 覆盖
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir --timeout 120 --retries 5 \
    -i ${PIP_INDEX_URL} -r requirements-serve.txt

# 代码和模型变化最频繁，放最后
COPY api/ ./api/
COPY model/model.onnx ./model/model.onnx

EXPOSE 8000

# --host 0.0.0.0：容器内必须监听所有网卡，否则宿主机连不进来
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
