FROM continuumio/miniconda3:23.10.0-1

WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git

COPY environment.yml .
RUN conda env create -f environment.yml

# Activate conda environment
SHELL ["conda", "run", "-n", "histo3.10", "/bin/bash", "-c"]
RUN pip install scikit-image

# Copy all files including .git directory
COPY . .

# Initialize and install submodules
RUN git submodule init && \
    git submodule update && \
    git submodule update --remote

# Install CellViT
WORKDIR /app/resources/CellViT
RUN git config --global --add safe.directory /app/resources/CellViT && \
    pip install -e .

# Install SimpleClick
WORKDIR /app/resources/SimpleClick
RUN git config --global --add safe.directory /app/resources/SimpleClick && \
    pip install -e .

WORKDIR /app
# Install the main package
RUN pip install -e .

# Expose the default Gradio port
EXPOSE 7860

# Create a startup script
RUN echo '#!/bin/bash' > /app/start.sh && \
    echo 'if [ "$1" = "bash" ]; then' >> /app/start.sh && \
    echo '    exec bash' >> /app/start.sh && \
    echo 'else' >> /app/start.sh && \
    echo '    export GRADIO_SERVER_NAME=0.0.0.0' >> /app/start.sh && \
    echo '    exec conda run --no-capture-output -n histo3.10 python scripts/app.py --model_dir models --model_name samhi --cellvit_model CellViT-256-x40.pth "$@"' >> /app/start.sh && \
    echo 'fi' >> /app/start.sh && \
    chmod +x /app/start.sh

ENTRYPOINT ["/app/start.sh"]