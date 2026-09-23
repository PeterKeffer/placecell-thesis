set -euo pipefail

if [[ -n "${PLACECELL_TMP_ROOT:-}" ]]; then
  return 0
fi

_placecell_pick_tmp_root() {
  local candidate="${SLURM_TMPDIR:-}"
  if [[ -n "${candidate}" && -w "${candidate}" ]]; then
    local avail_kb
    local avail_inodes
    avail_kb="$(df -Pk "${candidate}" 2>/dev/null | awk 'NR==2 {print $4}')"
    avail_inodes="$(df -Pi "${candidate}" 2>/dev/null | awk 'NR==2 {print $4}')"
    if [[ -n "${avail_kb}" && -n "${avail_inodes}" && "${avail_kb}" -ge 1048576 && "${avail_inodes}" -ge 1024 ]]; then
      echo "${candidate}"
      return
    fi
    echo "[placecell_research] SLURM_TMPDIR=${candidate} is too full (free_kb=${avail_kb:-unknown}, free_inodes=${avail_inodes:-unknown}); using runs/tmp" >&2
  fi
  echo "${PWD}/runs/tmp"
}

PLACECELL_TMP_ROOT="$(_placecell_pick_tmp_root)/placecell_${SLURM_JOB_ID:-manual}_${$}"
mkdir -p "${PLACECELL_TMP_ROOT}"
PLACECELL_TMP_LINK=""
for _placecell_link_base in /dev/shm /tmp "${XDG_RUNTIME_DIR:-}" "${SLURM_TMPDIR:-}"; do
  [[ -n "${_placecell_link_base}" && -w "${_placecell_link_base}" ]] || continue
  _placecell_link_candidate="${_placecell_link_base}/pc_${SLURM_JOB_ID:-manual}_${$}"
  if ln -sfn "${PLACECELL_TMP_ROOT}" "${_placecell_link_candidate}" 2>/dev/null; then
    PLACECELL_TMP_LINK="${_placecell_link_candidate}"
    break
  fi
done
if [[ -n "${PLACECELL_TMP_LINK}" ]]; then
  export TMPDIR="${PLACECELL_TMP_LINK}"
else
  echo "[placecell_research] WARNING: no writable short dir for the AF_UNIX link; using ${PLACECELL_TMP_ROOT}" >&2
  export TMPDIR="${PLACECELL_TMP_ROOT}"
fi
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"

PLACECELL_XDG_RUNTIME_DIR=""
placecell_cleanup_hpc_env() {
  if [[ -n "${PLACECELL_XDG_RUNTIME_DIR:-}" ]]; then
    rm -rf "${PLACECELL_XDG_RUNTIME_DIR}"
  fi
  if [[ -n "${PLACECELL_TMP_LINK:-}" ]]; then
    rm -f "${PLACECELL_TMP_LINK}"
  fi
  if [[ -n "${PLACECELL_TMP_ROOT:-}" ]]; then
    rm -rf "${PLACECELL_TMP_ROOT}"
  fi
}

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export WANDB_CACHE_DIR="${PLACECELL_TMP_ROOT}/wandb_cache"
export WANDB_DATA_DIR="${PLACECELL_TMP_ROOT}/wandb_data"
export XDG_CACHE_HOME="${PLACECELL_TMP_ROOT}/xdg_cache"
export MPLCONFIGDIR="${PLACECELL_TMP_ROOT}/matplotlib"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

echo "[placecell_research] host: $(hostname)"
echo "[placecell_research] TMPDIR=${TMPDIR} (-> $(readlink -f "${TMPDIR}" 2>/dev/null || echo "${TMPDIR}"))"
if command -v nvidia-smi >/dev/null 2>&1; then
  timeout 10s nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null \
    || echo "[placecell_research] nvidia-smi failed or timed out"
fi
