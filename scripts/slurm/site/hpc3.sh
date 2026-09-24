export http_proxy=http://rhn-proxy.rz.uos.de:3128
export https_proxy="${http_proxy}"

_placecell_shell_options="$(shopt -po nounset pipefail || true)"
case "$-" in *e*) _placecell_shell_options+=$'\nset -e' ;; esac
set +euo pipefail
if ! command -v spack >/dev/null 2>&1 && [[ -f /appl/spack/share/spack/setup-env.sh ]]; then
  source /appl/spack/share/spack/setup-env.sh
fi
if command -v spack >/dev/null 2>&1; then
  for _placecell_spack_spec in miniconda3@4.10.3 git@2.31.1; do
    spack load "${_placecell_spack_spec}" \
      || echo "[placecell_research] WARNING: spack load ${_placecell_spack_spec} failed" >&2
  done
  if [[ -z "${PLACECELL_EXTRA_LIBRARY_PATH:-}" ]]; then
    for _placecell_spack_spec in mesa-glu@9.0.1 mesa-glu@9.0.2 mesa-glu; do
      if spack find "${_placecell_spack_spec}" >/dev/null 2>&1; then
        spack load "${_placecell_spack_spec}"
        PLACECELL_EXTRA_LIBRARY_PATH="$(spack location -i "${_placecell_spack_spec}")/lib"
        break
      fi
    done
  fi
else
  echo "[placecell_research] WARNING: no spack; hpc3 provides conda, git and mesa-glu through it" >&2
fi
eval "${_placecell_shell_options}"
unset _placecell_shell_options _placecell_spack_spec

if [[ -z "${PLACECELL_EXTRA_LIBRARY_PATH:-}" && ( -f /usr/lib64/libGLU.so || -f /usr/lib64/libGLU.so.1 ) ]]; then
  PLACECELL_EXTRA_LIBRARY_PATH="/usr/lib64"
fi
if [[ -z "${PLACECELL_EXTRA_LIBRARY_PATH:-}" ]]; then
  echo "[placecell_research] WARNING: no libGLU found (spack mesa-glu or /usr/lib64); MiniWorld may fail to render" >&2
fi
export PLACECELL_EXTRA_LIBRARY_PATH
echo "[placecell_research] hpc3: proxy ${https_proxy}, conda $(command -v conda || echo '<none>'), libGLU from ${PLACECELL_EXTRA_LIBRARY_PATH:-<none>}"
