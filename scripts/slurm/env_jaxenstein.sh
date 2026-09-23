source "$(dirname "${BASH_SOURCE[0]}")/env_common.sh"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [[ "${PLACECELL_REQUESTED_GPUS:-1}" -eq 0 ]]; then
  export JAX_PLATFORMS=cpu
fi

python - <<'PY'
import os

import jax
import jaxenstein  # noqa: F401

devices = jax.devices()
print("[placecell_research] jax", jax.__version__, "devices", devices)
requested_gpus = int(os.environ.get("PLACECELL_REQUESTED_GPUS", "1") or "1")
has_gpu = any(getattr(device, "platform", "") in {"cuda", "gpu"} for device in devices)
if requested_gpus > 0 and not has_gpu:
    raise SystemExit("[placecell_research] ERROR: JAX sees no GPU device on this node.")
if requested_gpus == 0 and has_gpu:
    raise SystemExit("[placecell_research] ERROR: JAX claimed a GPU on a gpus=0 job.")
PY
echo "[placecell_research] JAXenstein environment ready"
