#!/bin/bash
# Restore IMLE-WM on a RunPod pod for revision r1 (main/revision/DESIGN_r1.md section 0). Runs ON THE POD.
# Needs /root/.config/rclone/rclone.conf with the user's read-only remote "gdrive-ro" (Drive is never written).
#   bash setup_pod.sh fetch <item> [<item> ...]   items: ckpt (tier1: code, runs, checkpoints), tier4 dataset names
#                                                   (bounce_div_k4, ...), tier5 names (push, pusht_b, pusht_det, bair)
#   bash setup_pod.sh env                          venv from ~/imle-wm/code/requirements.txt (push the code first)
set -euo pipefail
R=gdrive-ro:racedreamer-backup-2026-11
W=/workspace
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH UV_CACHE_DIR=$W/.uv-cache UV_HTTP_TIMEOUT=300
mkdir -p $W/nom/out $W/nom/big $W/nom/logs $W/imle-wm/logs $W/imle-wm/runs $W/imle-wm/data
ln -sfn $W/imle-wm $HOME/imle-wm
log() { echo "$(date -u +%H:%M:%SZ) $*"; }
tools() {
  command -v rsync >/dev/null && command -v unzip >/dev/null || { apt-get update -qq && apt-get install -y -qq rsync unzip >/dev/null; }
  if ! command -v rclone >/dev/null; then
    curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip
    rm -rf /tmp/rc && unzip -q /tmp/rc.zip -d /tmp/rc && install -m 755 /tmp/rc/rclone-*-linux-amd64/rclone /usr/local/bin/rclone
  fi
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
}
case "${1:-}" in
  fetch)
    shift; tools
    rclone lsf "$R" --max-depth 1 >/dev/null || { echo "rclone remote gdrive-ro is not usable"; exit 1; }
    [ -d $W/imle-wm/.hf ] || { log "hf cache"; rclone cat $R/tier6/hf-caches/imle-wm-hf.tar | tar -xf - -C $W; }
    for it in "$@"; do
      case "$it" in
        ckpt) [ -f $W/imle-wm/runs/e1_a2_imle16_s0/ckpt.pt ] || { log "tier1: code, runs, checkpoints"; rclone cat $R/tier1/t1_imle-wm.tgz | tar -xzf - -C $W; } ;;
        push|pusht_b|pusht_det|bair) [ -d $W/imle-wm/data/$it ] || { log "data $it (tier5)"; rclone copy $R/tier5/imle-wm-data/$it $W/imle-wm/data/$it --transfers 8 --multi-thread-streams 8; } ;;
        *) [ -d $W/imle-wm/data/$it ] || { log "data $it (tier4)"; rclone copy $R/tier4/imle-wm-data/$it $W/imle-wm/data/$it --transfers 8 --multi-thread-streams 8; } ;;
      esac
    done
    log "fetch done"; df -h $W | tail -1 ;;
  env)
    tools
    [ -f $W/imle-wm/code/requirements.txt ] || { echo "push main/code to ~/imle-wm/code first"; exit 1; }
    cat > $W/imle-wm/env.sh <<'ENV'
# source ~/imle-wm/env.sh
export IMLE=$HOME/imle-wm
export IMLE_DATA=$IMLE/data
export HF_HOME=$IMLE/.hf
export PYTHONPATH=$IMLE/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $IMLE/.venv/bin/activate
ENV
    cp $W/imle-wm/env.sh $W/imle-wm/code/env.sh
    if [ ! -x $W/imle-wm/.venv/bin/python ]; then
      log "venv (Python 3.10.12, original freeze)"
      cd $W/imle-wm && uv python install 3.10.12 && uv venv --python 3.10.12 .venv
      uv pip install --python .venv/bin/python -r code/requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match
    fi
    source $W/imle-wm/env.sh
    python -c "import torch, transformers, numpy; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0), '| transformers', transformers.__version__, '| numpy', numpy.__version__)"
    free -g | head -2; nproc; log "env done" ;;
  *) echo "usage: setup_pod.sh fetch <items...> | env"; exit 1 ;;
esac
