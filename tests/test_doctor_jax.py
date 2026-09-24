"""pc doctor reports JAX on a node without a GPU in one line, without the CUDA plugin traceback."""

from __future__ import annotations

import logging
import sys
import types

from placecell_research.launch.commands import doctor


def jax_whose_cuda_plugin_fails():
    def devices():
        try:
            raise RuntimeError("cuInit(0) failed: error 303")
        except RuntimeError:
            logging.getLogger("jax._src.xla_bridge").exception("Jax plugin configuration error")
        return [types.SimpleNamespace(platform="cpu")]

    return types.SimpleNamespace(__version__="0.7.1", devices=devices)


def test_cuda_plugin_error_becomes_one_line(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "jax", jax_whose_cuda_plugin_fails())
    monkeypatch.setattr(doctor, "_module_version", lambda name: None)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    doctor._check_jax(doctor._Report(), render=False)
    output = capsys.readouterr()
    assert "Traceback" not in output.out + output.err
    assert output.err == ""
    jax_line = "OK    jax            0.7.1; no GPU visible, so jax runs on CPU"
    assert jax_line in output.out.splitlines()


def test_cuda_plugin_error_is_named_when_a_gpu_is_visible(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "jax", jax_whose_cuda_plugin_fails())
    monkeypatch.setattr(doctor, "_module_version", lambda name: None)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    doctor._check_jax(doctor._Report(), render=False)
    jax_line = capsys.readouterr().out.splitlines()[0]
    assert jax_line.startswith("WARN") and jax_line.endswith("cuInit(0) failed: error 303")
