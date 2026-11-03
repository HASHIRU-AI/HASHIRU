#!/bin/zsh
# The trained splitter (plan/2026-11-02-finetuning-scope.md, §10), on the rented instance once the pilot is done: copy
# the data and the trainer, train on all of it, serve the merged model and then the base over an SSH
# tunnel, ask each for the 80 evaluation splits, and report. No .env is loaded; the cost is the instance's time.
#
#   SSH_KEY=~/Downloads/lambda bench/lambda/finetune.sh ubuntu@INSTANCE_IP     # then TERMINATE the instance
set -eu
HOST=${1:?usage: bench/lambda/finetune.sh ubuntu@INSTANCE_IP}
cd /Users/kunpai/Desktop/prototypes/HASHIRU
export PYTHONPATH=bench
S=bench/out/splitter
L=$S/logs
LOCAL=18000
SSHO=(-o IdentitiesOnly=yes -o ServerAliveInterval=30)
[[ -n ${SSH_KEY:-} ]] && SSHO+=(-i ${~SSH_KEY})
mkdir -p $L
[[ -s $S/train.jsonl ]] || uv run python bench/lambda/splitter_data.py
ssh $SSHO $HOST 'mkdir -p ~/splitter/data'
scp -q $SSHO $S/train.jsonl $S/val.jsonl ${HOST}:splitter/data/
scp -q $SSHO bench/lambda/train_splitter.py bench/lambda/splitter.yaml bench/lambda/serve.sh ${HOST}:splitter/
ssh $SSHO $HOST 'test -s splitter/data/train.jsonl && test -s splitter/train_splitter.py' || { echo "copy failed"; exit 1; }
# training packages go into vLLM's venv, which already has torch and transformers; the linear-attention kernels
# (flash-linear-attention) only make Qwen3.5's training faster, so a failed install is not an error
ssh $SSHO $HOST '~/.local/bin/uv pip install --python ~/vllm/bin/python peft accelerate omegaconf' > $L/install.log 2>&1
ssh $SSHO $HOST '~/.local/bin/uv pip install --python ~/vllm/bin/python flash-linear-attention' >> $L/install.log 2>&1 \
    || echo "flash-linear-attention did not install; training uses the slower PyTorch path"
echo "$(date +%T) training (all 3,000 examples, one epoch)"
ssh $SSHO $HOST 'cd splitter && ~/vllm/bin/python train_splitter.py' > $L/train.log 2>&1 \
    || { echo "training failed; see $L/train.log"; exit 1; }
scp -q $SSHO ${HOST}:splitter/run/log.json ${HOST}:splitter/run/env.json ${HOST}:splitter/run/config.yaml $S/

tunnel=0
serve() {  # $1: the model on the instance, $2: the name to serve it under. Returns once the server answers.
  ssh $SSHO -o ExitOnForwardFailure=yes -L ${LOCAL}:127.0.0.1:8000 $HOST "bash splitter/serve.sh $1 $2" \
      >> $L/serve-${2:t}.log 2>&1 &
  tunnel=$!
  for i in {1..120}; do
    curl -sf http://127.0.0.1:${LOCAL}/v1/models > /dev/null && return 0
    kill -0 $tunnel 2>/dev/null || break
    sleep 10
  done
  echo "vLLM did not come up for $2; see $L/serve-${2:t}.log"
  ssh $SSHO $HOST 'pkill -f "[v]llm serve"' || true
  exit 1
}
stop() {
  ssh $SSHO $HOST 'pkill -f "[v]llm serve"' || true
  kill $tunnel 2>/dev/null || true
  sleep 15
}
echo "$(date +%T) asking the trained splitter"
serve /home/ubuntu/splitter/run/merged splitter
uv run python bench/lambda/eval_splitter.py ask tuned splitter > $L/ask-tuned.log 2>&1
stop
echo "$(date +%T) asking the base model, without and with thinking"
serve Qwen/Qwen3.5-9B Qwen/Qwen3.5-9B
uv run python bench/lambda/eval_splitter.py ask base-nothink Qwen/Qwen3.5-9B > $L/ask-base-nothink.log 2>&1
uv run python bench/lambda/eval_splitter.py ask base-think Qwen/Qwen3.5-9B --think --max-tokens 8192 \
    > $L/ask-base-think.log 2>&1
stop
uv run python bench/lambda/eval_splitter.py report | tee $S/report.txt
echo "Splitter done. TERMINATE the instance in the Lambda dashboard now."
