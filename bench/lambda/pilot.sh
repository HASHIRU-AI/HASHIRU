#!/bin/zsh
# The open-weight pilot (plan/2026-11-02-finetuning-scope.md, §9 and §11): three Apache-2.0 models, each served in turn by vLLM
# on one rented Lambda GPU and reached over an SSH tunnel, calibrated on round 2's pool at levels 1, 3 and 5 (four jobs
# per cell, three attempts each: 432 attempts a model, in 40 shards at once). No .env is loaded, so nothing here can make a
# paid API call; what it costs is the instance's time. Tables resume, so a rerun picks up what is missing.
#
#   SSH_KEY=~/Downloads/lambda bench/lambda/pilot.sh ubuntu@INSTANCE_IP   # then TERMINATE the instance
#   PYTHONPATH=bench uv run python bench/lambda/pilot_report.py
# PILOT_MODELS="Qwen/Qwen3.5-27B" runs only the models named (space-separated), e.g. to finish after a failure.
set -eu
HOST=${1:?usage: bench/lambda/pilot.sh ubuntu@INSTANCE_IP}
cd /Users/kunpai/Desktop/prototypes/HASHIRU
export PYTHONPATH=bench
R=bench/out/r2
P=bench/out/pilot
L=$P/logs
LOCAL=18000  # the tunnel's laptop end; vLLM listens on the instance's 127.0.0.1:8000 only
SHARDS=40
SSHO=(-o IdentitiesOnly=yes -o ServerAliveInterval=30)
[[ -n ${SSH_KEY:-} ]] && SSHO+=(-i ${~SSH_KEY})
mkdir -p $L
# ${HOST} in braces: in zsh, $HOST:s... would be read as a modifier and scp would copy to a local file
scp -q $SSHO bench/lambda/serve.sh ${HOST}:serve.sh
ssh $SSHO $HOST 'test -s serve.sh' || { echo "serve.sh did not reach $HOST"; exit 1; }
typeset -A SHORT=(
  mistralai/Ministral-3-3B-Instruct-2512 ministral3b
  Qwen/Qwen3.5-9B qwen9b
  google/gemma-4-31B-it gemma31b
  Qwen/Qwen3.5-27B qwen27b
)
# Qwen3.5 27B is out of the default run (2026-11-02): at temperature 0 it would think for hours; name it to add it
MODELS=(${=PILOT_MODELS:-mistralai/Ministral-3-3B-Instruct-2512 Qwen/Qwen3.5-9B google/gemma-4-31B-it})
for model in $MODELS; do
  name=${SHORT[$model]}
  echo "$(date +%T) $name: starting vLLM (the first start installs it; a download and load can take 10+ minutes)"
  ssh $SSHO -o ExitOnForwardFailure=yes -L ${LOCAL}:127.0.0.1:8000 $HOST "bash serve.sh $model" \
      >> $L/serve-$name.log 2>&1 &
  tunnel=$!
  up=0
  for i in {1..180}; do  # up to 30 minutes
    if curl -sf http://127.0.0.1:$LOCAL/v1/models > /dev/null; then up=1; break; fi
    kill -0 $tunnel 2>/dev/null || break
    sleep 10
  done
  if (( ! up )); then
    echo "$name: vLLM did not come up; see $L/serve-$name.log. Stopping (the instance is still running)."
    kill $tunnel 2>/dev/null || true
    ssh $SSHO $HOST 'pkill -f "[v]llm serve"' || true
    exit 1
  fi
  echo "$(date +%T) $name: serving; calibrating"
  pids=()
  # 40 shards over all of the model's jobs, sharing one table: an attempt can take minutes (a thinking model at
  # temperature 0 writes up to the 8,192-token cap), so the work is spread wide; the cap, not the clock, ends an
  # attempt (--deadline 3600)
  for (( k = 0; k < SHARDS; k++ )); do
    # --max-usd counts list prices (what a provider would charge), not money spent: a shard stops past it
    uv run python -m purse calibrate --pool $R/pool.jsonl --spend --per-cell 4 --repeats 3 --levels 1,3,5 \
        --options hosted_vllm/$model --endpoint hosted_vllm/$model=http://127.0.0.1:$LOCAL/v1 \
        --shard $k/$SHARDS --deadline 3600 --table $P/t-$name.jsonl --max-usd 2 --port $((9600 + k)) \
        > $L/$name-s$k.log 2>&1 &
    pids+=($!)
  done
  failed=0
  for pid in $pids; do wait $pid || failed=$((failed + 1)); done
  ssh $SSHO $HOST 'pkill -f "[v]llm serve"' >> $L/serve-$name.log 2>&1 || true
  kill $tunnel 2>/dev/null || true
  sleep 15  # let the GPU memory free before the next model loads
  echo "$(date +%T) $name: done ($failed of ${#pids} shards stopped early; logs in $L)"
done
ssh $SSHO $HOST '~/vllm/bin/python -c "import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__)"; nvidia-smi -L' \
    > $P/instance.txt 2>&1 || true
echo "Pilot calls done. TERMINATE the instance in the Lambda dashboard now, then run:"
echo "  PYTHONPATH=bench uv run python bench/lambda/pilot_report.py"
