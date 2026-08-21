#!/usr/bin/env bash
# Bring the 27B test run back up after a reboot. Run from Windows with:
#   wsl -d Ubuntu -- bash -lc "bash '/mnt/c/Users/imarg/syncthing_shared/Coding Projects/MuseAI/.resume-27b-run.sh'"
#
# Two things this sets up that are NOT the default on this host:
#
# 1. Ollama on :11435, owned by imarg. The systemd ollama.service runs as
#    User=ollama and therefore reads /usr/share/ollama/.ollama/models, which is
#    empty; the real 61 GB store is /home/imarg/.ollama/models. We cannot stop
#    that service without a sudo password, so we run a second server alongside
#    it. config.yaml points at 11435 for exactly this reason.
# 2. OLLAMA_CONTEXT_LENGTH=16384, matching endpoint.context_window. Ollama's /v1
#    endpoint silently discards extra_body.options.num_ctx, so this env var is
#    the only lever that binds.
set -u
PROJ="/mnt/c/Users/imarg/syncthing_shared/Coding Projects/MuseAI"
cd "$PROJ" || exit 1

echo "== GPU check =="
if ! nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null; then
  echo "  GPU STILL UNAVAILABLE in WSL. Check 'nvidia-smi' on Windows first --"
  echo "  'GPU is lost' there means the host has not recovered the card yet."
  exit 1
fi

echo "== ollama on 11435 =="
pkill -f 'ollama serve' -U "$(id -u)" 2>/dev/null
sleep 3
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_KEEP_ALIVE=30m OLLAMA_CONTEXT_LENGTH=16384 \
  setsid nohup ollama serve </dev/null >/tmp/ollama-imarg.log 2>&1 &
disown
for i in $(seq 1 20); do
  curl -s --max-time 3 http://127.0.0.1:11435/api/version >/dev/null 2>&1 && break
  sleep 2
done
n=$(curl -s --max-time 20 http://127.0.0.1:11435/api/tags \
    | python3 -c "import json,sys;print(len(json.load(sys.stdin)['models']))" 2>/dev/null)
echo "  models visible: ${n:-0}  (expect 12; 0 means it found the wrong store)"

echo "== clear previous run =="
rm -f data/museai.db data/museai.db-wal data/museai.db-shm data/chat.jsonl data/events.jsonl
rm -f data/drafts/* data/output/*.md
: > logs/fsm.log; : > logs/llm_io.log; : > logs/app.jsonl
echo "  db, drafts, exports and logs cleared"

echo "== MuseAI server =="
pkill -f 'python run.py' -U "$(id -u)" 2>/dev/null
sleep 2
MUSEAI_API_KEY=ollama setsid nohup uv run python run.py </dev/null >/tmp/museai-server.log 2>&1 &
disown
for i in $(seq 1 40); do
  if curl -s --max-time 5 http://127.0.0.1:8000/status >/dev/null 2>&1; then
    echo "  MuseAI up on http://127.0.0.1:8000"
    curl -s --max-time 15 http://127.0.0.1:8000/status \
      | python3 -c "import json,sys;d=json.load(sys.stdin);print('  status:',d['status'],'| endpoint:',d['endpoint']['model_name'])"
    exit 0
  fi
  sleep 3
done
echo "  FAILED to come up"; tail -30 /tmp/museai-server.log
