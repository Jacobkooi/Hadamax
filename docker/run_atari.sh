docker run -it --rm --gpus=all --ipc=host \
  -v "$(pwd)":/app \
  -w /app \
  --name pqn-atari-hadamax \
  pqn-atari-hadamax \
  bash