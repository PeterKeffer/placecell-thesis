#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_INSTALL_PREFIX="${PLACECELL_CONDA_PREFIX:-${HOME}/miniforge3}"
ENV_NAME="${PLACECELL_ENV_NAME:-placecell}"
PYTHON_VERSION="3.12"
GPU_MODE="auto"
EXTRAS="rl,jax,dev"
TORCH_INDEX_URL=""
DRY_RUN=0
RUN_DOCTOR=1

usage() {
  cat <<'USAGE'
Usage: scripts/setup_env.sh [options]

Creates your own conda environment for this repository and checks it with pc doctor.
If no conda or mamba is found, Miniforge is installed into --prefix first.

  --prefix DIR        where to install Miniforge if no conda is found
                      (default: $PLACECELL_CONDA_PREFIX or ~/miniforge3)
  --env NAME          environment name (default: $PLACECELL_ENV_NAME or placecell)
  --gpu MODE          auto, cuda or cpu (auto: cuda on Linux with nvidia-smi or sbatch,
                      cpu elsewhere; macOS always uses the default build with MPS)
  --torch-index URL   pip index for torch, e.g. https://download.pytorch.org/whl/cu128
  --no-doctor         skip the final pc doctor check
  --dry-run           print every step without running it
  -h, --help          this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) CONDA_INSTALL_PREFIX="$2"; shift 2 ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    --gpu) GPU_MODE="$2"; shift 2 ;;
    --torch-index) TORCH_INDEX_URL="$2"; shift 2 ;;
    --no-doctor) RUN_DOCTOR=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "$@"
  fi
}

step() {
  echo
  echo "== $*"
}

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
case "${OS_NAME}" in
  Darwin|Linux) ;;
  *) echo "Unsupported system ${OS_NAME}; use macOS or Linux." >&2; exit 1 ;;
esac

if [[ "${GPU_MODE}" == "auto" ]]; then
  if [[ "${OS_NAME}" == "Linux" ]] && { command -v nvidia-smi >/dev/null 2>&1 || command -v sbatch >/dev/null 2>&1; }; then
    GPU_MODE="cuda"
  else
    GPU_MODE="cpu"
  fi
fi
case "${GPU_MODE}" in
  cuda|cpu) ;;
  *) echo "--gpu must be auto, cuda or cpu, not ${GPU_MODE}" >&2; exit 2 ;;
esac

step "system: ${OS_NAME} ${ARCH_NAME}, torch/jax build: ${GPU_MODE}, repo: ${REPO_ROOT}"

step "1. find conda or mamba"
CONDA_BIN=""
for candidate in "${CONDA_EXE:-}" "$(command -v conda 2>/dev/null || true)" "${CONDA_INSTALL_PREFIX}/bin/conda" "$(command -v mamba 2>/dev/null || true)"; do
  if [[ -n "${candidate}" && -x "${candidate}" ]] && "${candidate}" --version >/dev/null 2>&1; then
    CONDA_BIN="${candidate}"
    break
  fi
done
if [[ -n "${CONDA_BIN}" ]]; then
  echo "using ${CONDA_BIN}"
  CONDA_BASE="$("${CONDA_BIN}" info --base)"
else
  echo "no conda found; installing Miniforge into ${CONDA_INSTALL_PREFIX}"
  INSTALLER_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-${OS_NAME}-${ARCH_NAME}.sh"
  INSTALLER_PATH="${TMPDIR:-/tmp}/Miniforge3-${OS_NAME}-${ARCH_NAME}.sh"
  run curl -fsSL -o "${INSTALLER_PATH}" "${INSTALLER_URL}"
  run bash "${INSTALLER_PATH}" -b -p "${CONDA_INSTALL_PREFIX}"
  run rm -f "${INSTALLER_PATH}"
  CONDA_BIN="${CONDA_INSTALL_PREFIX}/bin/conda"
  CONDA_BASE="${CONDA_INSTALL_PREFIX}"
fi
ENV_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"
PYTHON_BIN="${ENV_PREFIX}/bin/python"

step "2. create the environment ${ENV_PREFIX}"
if [[ -x "${PYTHON_BIN}" ]]; then
  echo "environment exists; updating it"
else
  run "${CONDA_BIN}" create -y -p "${ENV_PREFIX}" -c conda-forge "python=${PYTHON_VERSION}" pip
fi
run "${PYTHON_BIN}" -m pip install --upgrade pip

step "3. PyTorch (${GPU_MODE})"
TORCH_ARGS=(torch)
if [[ -n "${TORCH_INDEX_URL}" ]]; then
  TORCH_ARGS+=(--index-url "${TORCH_INDEX_URL}")
elif [[ "${OS_NAME}" == "Linux" && "${GPU_MODE}" == "cpu" ]]; then
  TORCH_ARGS+=(--index-url https://download.pytorch.org/whl/cpu)
fi
run "${PYTHON_BIN}" -m pip install "${TORCH_ARGS[@]}"

step "4. this package (editable, extras ${EXTRAS}), MiniWorld, pyglet, JAX and JAXenstein"
PACKAGE_ARGS=(-e "${REPO_ROOT}[${EXTRAS}]")
if [[ "${OS_NAME}" == "Linux" && "${GPU_MODE}" == "cuda" ]]; then
  PACKAGE_ARGS+=("jax[cuda12]>=0.6.2")
fi
run "${PYTHON_BIN}" -m pip install "${PACKAGE_ARGS[@]}"

step "5. OpenGL/EGL for headless MiniWorld"
if [[ "${OS_NAME}" == "Darwin" ]]; then
  echo "macOS renders through the window system: keep the display awake (caffeinate -d -i)."
else
  EGL_FOUND=""
  for library in /usr/lib64/libEGL.so.1 /usr/lib/x86_64-linux-gnu/libEGL.so.1 /usr/lib/aarch64-linux-gnu/libEGL.so.1; do
    if [[ -f "${library}" ]]; then
      EGL_FOUND="${library}"
      break
    fi
  done
  if [[ -n "${EGL_FOUND}" ]]; then
    echo "system libEGL: ${EGL_FOUND} (GPU jobs use the NVIDIA EGL vendor through it)"
  else
    echo "WARNING: no system libEGL.so.1. Ask for libglvnd (libEGL) on the GPU nodes or set PLACECELL_EGL_LIBRARY."
  fi
  echo "CPU-only MiniWorld jobs need software Mesa: set PLACECELL_MESA_PREFIX (see README)."
fi

step "6. self-check"
if [[ "${RUN_DOCTOR}" -eq 1 ]]; then
  DOCTOR_ARGS=()
  if [[ "${OS_NAME}" == "Linux" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
    DOCTOR_ARGS+=(--no-render)
  fi
  (cd "${REPO_ROOT}" && run "${PYTHON_BIN}" -m placecell_research.launch.cli doctor ${DOCTOR_ARGS[@]+"${DOCTOR_ARGS[@]}"})
fi

step "done"
cat <<DONE
Activate the environment with:
  source "${CONDA_BASE}/etc/profile.d/conda.sh" && conda activate "${ENV_PREFIX}"
Python for SLURM jobs and for remote.python in the user file on your laptop:
  ${PYTHON_BIN}
DONE
