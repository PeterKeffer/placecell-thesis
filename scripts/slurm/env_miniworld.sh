source "$(dirname "${BASH_SOURCE[0]}")/env_common.sh"

export PYOPENGL_PLATFORM=egl
export PYGLET_HEADLESS=1
export MINIWORLD_HEADLESS=1
export DISPLAY=

if [[ "${PLACECELL_REQUESTED_GPUS:-0}" -gt 0 ]]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
  PLACECELL_EGL_LIBRARY="${PLACECELL_EGL_LIBRARY:-}"
  if [[ -z "${PLACECELL_EGL_LIBRARY}" ]]; then
    for _placecell_egl_candidate in /usr/lib64/libEGL.so.1 /usr/lib/x86_64-linux-gnu/libEGL.so.1; do
      if [[ -f "${_placecell_egl_candidate}" ]]; then
        PLACECELL_EGL_LIBRARY="${_placecell_egl_candidate}"
        break
      fi
    done
  fi
  if [[ -z "${PLACECELL_EGL_LIBRARY}" || ! -f "${PLACECELL_EGL_LIBRARY}" ]]; then
    echo "[placecell_research] ERROR: libEGL.so.1 not found; set PLACECELL_EGL_LIBRARY" >&2
    exit 1
  fi
  EGL_LINK_DIR="${PLACECELL_TMP_ROOT}/egl_link"
  mkdir -p "${EGL_LINK_DIR}"
  ln -sf "${PLACECELL_EGL_LIBRARY}" "${EGL_LINK_DIR}/libEGL.so"
  export LD_LIBRARY_PATH="${EGL_LINK_DIR}:$(dirname "${PLACECELL_EGL_LIBRARY}")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
  if [[ -z "${PLACECELL_MESA_PREFIX:-}" || ! -d "${PLACECELL_MESA_PREFIX}/lib/dri" ]]; then
    echo "[placecell_research] ERROR: a CPU MiniWorld job needs software Mesa; set PLACECELL_MESA_PREFIX" >&2
    exit 1
  fi
  export __EGL_VENDOR_LIBRARY_FILENAMES="${PLACECELL_MESA_PREFIX}/share/glvnd/egl_vendor.d/50_mesa.json"
  export LIBGL_DRIVERS_PATH="${PLACECELL_MESA_PREFIX}/lib/dri"
  export LD_LIBRARY_PATH="${PLACECELL_MESA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export LP_NUM_THREADS=1
  PLACECELL_XDG_RUNTIME_DIR="/dev/shm/pc_xdg_${SLURM_JOB_ID:-manual}_$$"
  export XDG_RUNTIME_DIR="${PLACECELL_XDG_RUNTIME_DIR}"
  mkdir -p "${XDG_RUNTIME_DIR}"
  chmod 700 "${XDG_RUNTIME_DIR}"
fi

python -c "import miniworld" || {
  echo "[placecell_research] ERROR: miniworld is not importable in this environment" >&2
  exit 1
}
echo "[placecell_research] MiniWorld environment ready (PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM})"
