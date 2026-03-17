#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if ! python3.12 --version >/dev/null 2>&1; then
  uv python install 3.12
fi

uv venv --python 3.12 --clear .venv
source .venv/bin/activate

set +e
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "cu130 nightly install failed; falling back to cu128 nightly." >&2
  uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
fi

uv pip install -e .

python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
