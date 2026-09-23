_placecell_enable_spack() {
  if command -v spack >/dev/null 2>&1; then
    return 0
  fi
  if [[ ! -f /appl/spack/share/spack/setup-env.sh ]]; then
    return 1
  fi
  set +eu
  source /appl/spack/share/spack/setup-env.sh
  set -eu
  command -v spack >/dev/null 2>&1
}

if [[ "${PLACECELL_ENVIRONMENT_KIND:-}" == "miniworld" && "${PLACECELL_REQUESTED_GPUS:-0}" -gt 0 ]]; then
  export PLACECELL_EGL_LIBRARY="${PLACECELL_EGL_LIBRARY:-/usr/lib64/libEGL.so.1}"
  PLACECELL_EXTRA_LIBRARY_PATH="${PLACECELL_EXTRA_LIBRARY_PATH:-}"
  if [[ -z "${PLACECELL_EXTRA_LIBRARY_PATH}" ]] && _placecell_enable_spack; then
    for _placecell_glu_spec in mesa-glu@9.0.1 mesa-glu@9.0.2 mesa-glu; do
      if spack find "${_placecell_glu_spec}" >/dev/null 2>&1; then
        spack load "${_placecell_glu_spec}" >/dev/null 2>&1 || true
        PLACECELL_EXTRA_LIBRARY_PATH="$(spack location -i "${_placecell_glu_spec}")/lib"
        break
      fi
    done
  fi
  if [[ -z "${PLACECELL_EXTRA_LIBRARY_PATH}" && ( -f /usr/lib64/libGLU.so || -f /usr/lib64/libGLU.so.1 ) ]]; then
    PLACECELL_EXTRA_LIBRARY_PATH="/usr/lib64"
  fi
  if [[ -z "${PLACECELL_EXTRA_LIBRARY_PATH}" ]]; then
    echo "[placecell_research] WARNING: no libGLU found (spack mesa-glu or /usr/lib64); MiniWorld may fail to render" >&2
  fi
  export PLACECELL_EXTRA_LIBRARY_PATH
  echo "[placecell_research] hpc3: libGLU from ${PLACECELL_EXTRA_LIBRARY_PATH:-<none>}"
fi
