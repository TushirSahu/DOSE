FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /workspace/DOSE

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]
