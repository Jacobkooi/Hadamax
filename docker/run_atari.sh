docker run -it --rm --gpus=all --ipc=host \
 -v $(pwd):/app \
 -w /app \
 --name "pqn-${gpu//,/-}" \
 pqn-atari \
 bash
