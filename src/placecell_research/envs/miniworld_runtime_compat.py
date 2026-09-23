"""MiniWorld runtime compatibility helpers."""

from __future__ import annotations

import os as _os
import sys as _sys

if _sys.platform.startswith("linux"):
    _os.environ.setdefault("PYGLET_HEADLESS", "1")
    _os.environ.setdefault("MINIWORLD_HEADLESS", "1")
    _os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    _os.environ.setdefault(
        "__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    )
    _os.environ.setdefault("DISPLAY", "")

import ctypes
import ctypes.util
import os
import sys
from ctypes import c_float, c_int, c_uint
from typing import Any

import numpy as np

_EGL_PRELOADED = False
_FIND_LIBRARY_PATCHED = False


def _preload_egl_from_ld_library_path() -> bool:
    """Preload libEGL.so from LD_LIBRARY_PATH before pyglet tries to find it."""
    global _EGL_PRELOADED
    if _EGL_PRELOADED:
        return True

    if sys.platform != "linux":
        _EGL_PRELOADED = True
        return True

    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if not ld_library_path:
        return False

    for directory in ld_library_path.split(":"):
        if not directory:
            continue

        for lib_name in ("libEGL.so", "libEGL.so.1"):
            lib_path = os.path.join(directory, lib_name)
            if os.path.exists(lib_path):
                try:
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    _EGL_PRELOADED = True
                    return True
                except OSError:
                    continue

    return False


def _patch_find_library_for_ld_library_path() -> None:
    """Patch ctypes.util.find_library to check LD_LIBRARY_PATH first on Linux."""
    global _FIND_LIBRARY_PATCHED
    if _FIND_LIBRARY_PATCHED:
        return

    if sys.platform != "linux":
        _FIND_LIBRARY_PATCHED = True
        return

    original_find_library = ctypes.util.find_library

    def patched_find_library(name: str) -> str | None:
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        if ld_library_path:
            lib_names = [f"lib{name}.so", f"lib{name}.so.1", f"lib{name}.so.0"]
            for directory in ld_library_path.split(":"):
                if not directory:
                    continue
                for lib_name in lib_names:
                    lib_path = os.path.join(directory, lib_name)
                    if os.path.exists(lib_path):
                        return lib_path

        return original_find_library(name)

    ctypes.util.find_library = patched_find_library
    _FIND_LIBRARY_PATCHED = True


_patch_find_library_for_ld_library_path()
_preload_egl_from_ld_library_path()

_TEXTURE_LOADER_PATCHED = False
_COLOR_PALETTE_PATCHED = False
_TEXTURE_FALLBACK_PATCHED = False
_MESH_FALLBACK_PATCHED = False
_GL_LEGACY_PATCHED = False
_GL_API_CACHE: dict[str, Any] | None = None
_GL_MODULES: tuple[Any, ...] | None = None
_STDERR_FILTER_INSTALLED = False
_STDOUT_FILTER_INSTALLED = False

LEGACY_GL_CONSTANTS: dict[str, int] = {
    "GL_AMBIENT": 0x1200,
    "GL_AMBIENT_AND_DIFFUSE": 0x1602,
    "GL_ANY_SAMPLES_PASSED": 0x8C2F,
    "GL_COLOR_ATTACHMENT0": 0x8CE0,
    "GL_COLOR_BUFFER_BIT": 0x00004000,
    "GL_COLOR_MATERIAL": 0x0B57,
    "GL_COMPILE": 0x1300,
    "GL_CONSTANT_ATTENUATION": 0x1207,
    "GL_CULL_FACE": 0x0B44,
    "GL_DEPTH_ATTACHMENT": 0x8D00,
    "GL_DEPTH_BUFFER_BIT": 0x00000100,
    "GL_DEPTH_COMPONENT": 0x1902,
    "GL_DEPTH_COMPONENT16": 0x81A5,
    "GL_DEPTH_TEST": 0x0B71,
    "GL_DIFFUSE": 0x1201,
    "GL_DRAW_FRAMEBUFFER": 0x8CA9,
    "GL_FRAMEBUFFER": 0x8D40,
    "GL_FRAMEBUFFER_COMPLETE": 0x8CD5,
    "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT": 0x8CD6,
    "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER": 0x8CDB,
    "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS": 0x8DA8,
    "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT": 0x8CD7,
    "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE": 0x8D56,
    "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER": 0x8CDC,
    "GL_FRAMEBUFFER_UNDEFINED": 0x8219,
    "GL_FRAMEBUFFER_UNSUPPORTED": 0x8CDD,
    "GL_FRONT_AND_BACK": 0x0408,
    "GL_GENERATE_MIPMAP_HINT": 0x8192,
    "GL_LIGHT0": 0x4000,
    "GL_LIGHT1": 0x4001,
    "GL_LIGHT2": 0x4002,
    "GL_LIGHT3": 0x4003,
    "GL_LIGHT4": 0x4004,
    "GL_LIGHT5": 0x4005,
    "GL_LIGHT6": 0x4006,
    "GL_LIGHT7": 0x4007,
    "GL_LIGHTING": 0x0B50,
    "GL_LINEAR": 0x2601,
    "GL_LINEAR_ATTENUATION": 0x1208,
    "GL_LINEAR_MIPMAP_LINEAR": 0x2703,
    "GL_LINES": 0x0001,
    "GL_LINE_STRIP": 0x0003,
    "GL_MODELVIEW": 0x1700,
    "GL_MULTISAMPLE": 0x809D,
    "GL_NEAREST": 0x2600,
    "GL_NICEST": 0x1102,
    "GL_PACK_ALIGNMENT": 0x0D05,
    "GL_POLYGON": 0x0009,
    "GL_POSITION": 0x1203,
    "GL_PROJECTION": 0x1701,
    "GL_QUADRATIC_ATTENUATION": 0x1209,
    "GL_QUADS": 0x0007,
    "GL_QUERY_RESULT": 0x8866,
    "GL_READ_FRAMEBUFFER": 0x8CA8,
    "GL_RENDERBUFFER": 0x8D41,
    "GL_RGB": 0x1907,
    "GL_RGBA": 0x1908,
    "GL_RGBA32F": 0x8814,
    "GL_SMOOTH": 0x1D01,
    "GL_SPOT_CUTOFF": 0x1206,
    "GL_SPOT_DIRECTION": 0x1204,
    "GL_SPOT_EXPONENT": 0x1205,
    "GL_TEXTURE_2D": 0x0DE1,
    "GL_TEXTURE_2D_MULTISAMPLE": 0x9100,
    "GL_TEXTURE_MAG_FILTER": 0x2800,
    "GL_TEXTURE_MIN_FILTER": 0x2801,
    "GL_TRIANGLES": 0x0004,
    "GL_UNSIGNED_BYTE": 0x1401,
    "GL_UNSIGNED_SHORT": 0x1403,
}

LEGACY_GL_TYPES: dict[str, Any] = {
    "GLfloat": c_float,
    "GLint": c_int,
    "GLuint": c_uint,
}

LEGACY_GL_FUNCTIONS: tuple[str, ...] = (
    "glBegin",
    "glBeginQuery",
    "glBindFramebuffer",
    "glBindRenderbuffer",
    "glBindTexture",
    "glBlitFramebuffer",
    "glCallList",
    "glCheckFramebufferStatus",
    "glClear",
    "glClearColor",
    "glClearDepth",
    "glColor3f",
    "glColorMaterial",
    "glDeleteLists",
    "glDeleteQueries",
    "glDisable",
    "glEnable",
    "glEnd",
    "glEndList",
    "glEndQuery",
    "glFlush",
    "glFramebufferRenderbuffer",
    "glFramebufferTexture2D",
    "glGenFramebuffers",
    "glGenQueries",
    "glGenRenderbuffers",
    "glGenTextures",
    "glGenerateMipmap",
    "glGetIntegerv",
    "glGetQueryObjectuiv",
    "glHint",
    "glLightf",
    "glLightfv",
    "glLoadIdentity",
    "glLoadMatrixf",
    "glMatrixMode",
    "glNewList",
    "glNormal3f",
    "glOrtho",
    "glPixelStorei",
    "glPopMatrix",
    "glPushMatrix",
    "glReadPixels",
    "glRenderbufferStorage",
    "glRenderbufferStorageMultisample",
    "glRotatef",
    "glScalef",
    "glShadeModel",
    "glTexCoord2f",
    "glTexImage2D",
    "glTexImage2DMultisample",
    "glTexParameteri",
    "glTranslatef",
    "glVertex3f",
    "glViewport",
    "glEnable",
    "glDisable",
    "glLightModelf",
    "glMaterialf",
    "glMaterialfv",
    "glLightModelfv",
    "glLightModeli",
    "glLightModeliv",
    "glLightf",
    "glLightfv",
    "glMateriali",
    "glMaterialiv",
)

OPTIONAL_NOOPS: dict[str, Any] = {
    "glDeleteLists": lambda *_: None,
    "glNewList": lambda *_: None,
    "glEndList": lambda *_: None,
    "glCallList": lambda *_: None,
    "glGenLists": lambda *_: 0,
    "glLightfv": lambda *_: None,
    "glLightf": lambda *_: None,
    "glColorMaterial": lambda *_: None,
    "glShadeModel": lambda *_: None,
    "glEnable": lambda *_: None,
    "glDisable": lambda *_: None,
    "glLightModelf": lambda *_: None,
    "glLightModelfv": lambda *_: None,
    "glLightModeli": lambda *_: None,
    "glLightModeliv": lambda *_: None,
}


def ensure_miniworld_runtime_compatibility() -> None:
    """Apply the minimal runtime patches once."""
    _install_framebuffer_warning_filter()
    _install_logging_warning_filter()
    _disable_pyglet_shadow_window_on_macos()
    _disable_pyglet_msaa_requests()
    _ensure_texture_loader_is_core_profile_safe()
    _ensure_extended_color_palette()
    fallbacks_disabled = bool(
        os.environ.get("MINIWORLD_DISABLE_FALLBACKS")
        or os.environ.get("NEUROCELLS_DISABLE_MINIWORLD_FALLBACKS")
    )
    if not fallbacks_disabled:
        _ensure_texture_fallbacks()
        _ensure_mesh_fallbacks()


def _disable_pyglet_shadow_window_on_macos() -> None:
    """Avoid Pyglet's import-time shadow window when MiniWorld imports pyglet.gl."""
    if sys.platform != "darwin":
        return
    try:
        import pyglet

        pyglet.options["shadow_window"] = False
    except Exception:
        pass


def _install_framebuffer_warning_filter() -> None:
    """Silence noisy MiniWorld/pyglet framebuffer fallbacks without hiding other output."""
    global _STDERR_FILTER_INSTALLED, _STDOUT_FILTER_INSTALLED
    if _STDERR_FILTER_INSTALLED and _STDOUT_FILTER_INSTALLED:
        return

    import io
    import sys

    substrings = (
        "Falling back to num_samples",
        "non-multisampled frame buffer",
    )

    class _FilteredStream(io.TextIOBase):
        def __init__(self, wrapped_stream):
            self._wrapped = wrapped_stream
            self._pending = ""

        @property
        def encoding(self):
            return getattr(self._wrapped, "encoding", "utf-8")

        @property
        def errors(self):
            return getattr(self._wrapped, "errors", "replace")

        @property
        def newlines(self):
            return getattr(self._wrapped, "newlines", None)

        @property
        def buffer(self):
            return getattr(self._wrapped, "buffer", None)

        def writable(self):
            return True

        def _coerce_text(self, text):
            if isinstance(text, bytes):
                return text.decode(self.encoding or "utf-8", errors=self.errors or "replace")
            return str(text)

        def write(self, text):
            if not text:
                return 0
            normalized_text = self._coerce_text(text)
            self._pending += normalized_text
            while "\n" in self._pending:
                line, remainder = self._pending.split("\n", 1)
                self._pending = remainder
                if any(substring in line for substring in substrings):
                    continue
                self._wrapped.write(line + "\n")
            return len(text)

        def flush(self):
            if self._pending:
                if not any(substring in self._pending for substring in substrings):
                    self._wrapped.write(self._pending)
                self._pending = ""
            return self._wrapped.flush()

        def fileno(self):
            return self._wrapped.fileno() if hasattr(self._wrapped, "fileno") else -1

        def isatty(self):
            return self._wrapped.isatty() if hasattr(self._wrapped, "isatty") else False

    if not _STDERR_FILTER_INSTALLED:
        sys.stderr = _FilteredStream(sys.stderr)  # type: ignore
        _STDERR_FILTER_INSTALLED = True
    if not _STDOUT_FILTER_INSTALLED:
        sys.stdout = _FilteredStream(sys.stdout)  # type: ignore
        _STDOUT_FILTER_INSTALLED = True


def _install_logging_warning_filter() -> None:
    """Drop pyglet/MiniWorld framebuffer warnings emitted via the logging module."""
    import logging

    targets = ("pyglet", "pyglet.window", "pyglet.app", "pyglet.gl")
    for name in targets:
        logger = logging.getLogger(name)
        if logger.level > logging.WARNING:
            continue
        logger.setLevel(logging.ERROR)


def _disable_pyglet_msaa_requests() -> None:
    if os.environ.get("NEUROCELLS_PYGLET_DISABLE_MSAA", "1") == "0":
        return
    try:
        import pyglet

        pyglet.options["samples"] = 0
        pyglet.options["sample_buffers"] = 0
    except Exception:
        pass


def _get_gl_modules() -> tuple[Any, ...]:
    global _GL_MODULES
    if _GL_MODULES is None:
        modules: list[Any] = []
        try:
            from pyglet import gl as pyglet_gl

            modules.append(pyglet_gl)
            glu_mod = getattr(pyglet_gl, "glu", None)
            if glu_mod is not None:
                modules.append(glu_mod)
        except Exception:
            pass

        for name in ("OpenGL.GL", "OpenGL.GLU"):
            try:
                modules.append(__import__(name, fromlist=["*"]))
            except ModuleNotFoundError:
                continue
        _GL_MODULES = tuple(modules)
    return _GL_MODULES


def _lookup_gl_symbol(name: str) -> Any | None:
    if name in OPTIONAL_NOOPS:
        return OPTIONAL_NOOPS[name]
    for module in _get_gl_modules():
        value = getattr(module, name, None)
        if value is not None:
            return value
    if name in LEGACY_GL_CONSTANTS:
        return LEGACY_GL_CONSTANTS[name]
    if name in LEGACY_GL_TYPES:
        return LEGACY_GL_TYPES[name]
    return None


def _resolve_gl_api() -> dict[str, Any]:
    global _GL_API_CACHE
    if _GL_API_CACHE is None:
        required_constants = [
            "GL_GENERATE_MIPMAP_HINT",
            "GL_LINEAR",
            "GL_LINEAR_MIPMAP_LINEAR",
            "GL_NICEST",
            "GL_RGB",
            "GL_RGBA",
            "GL_TEXTURE_2D",
            "GL_TEXTURE_MAG_FILTER",
            "GL_TEXTURE_MIN_FILTER",
            "GL_UNSIGNED_BYTE",
        ]
        required_functions = [
            "glBindTexture",
            "glGenerateMipmap",
            "glHint",
            "glTexImage2D",
            "glTexParameteri",
        ]
        cache: dict[str, Any] = {}
        for const in required_constants:
            value = _lookup_gl_symbol(const)
            if value is None:
                raise RuntimeError(f"Missing required OpenGL constant '{const}'.")
            cache[const] = value
        for func in required_functions:
            value = _lookup_gl_symbol(func)
            if value is None:
                raise RuntimeError(f"Missing required OpenGL function '{func}'.")
            cache[func] = value
        _GL_API_CACHE = cache
    return _GL_API_CACHE


def _ensure_texture_loader_is_core_profile_safe() -> None:
    global _TEXTURE_LOADER_PATCHED
    if _TEXTURE_LOADER_PATCHED:
        return

    try:
        import miniworld.opengl as miniworld_gl  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("miniworld must be installed before applying patches.") from exc

    from pyglet import image

    gl_api = _resolve_gl_api()

    @classmethod
    def _patched_load(cls, texture_path: str):
        img = image.load(texture_path)
        texture = img.get_texture()
        target = getattr(texture, "target", gl_api["GL_TEXTURE_2D"])
        gl_api["glBindTexture"](target, texture.id)
        gl_api["glTexImage2D"](
            target,
            0,
            gl_api["GL_RGB"],
            img.width,
            img.height,
            0,
            gl_api["GL_RGBA"],
            gl_api["GL_UNSIGNED_BYTE"],
            img.get_image_data().get_data("RGBA", img.width * 4),
        )
        gl_api["glHint"](gl_api["GL_GENERATE_MIPMAP_HINT"], gl_api["GL_NICEST"])
        gl_api["glGenerateMipmap"](target)
        gl_api["glTexParameteri"](target, gl_api["GL_TEXTURE_MAG_FILTER"], gl_api["GL_LINEAR"])
        gl_api["glTexParameteri"](
            target, gl_api["GL_TEXTURE_MIN_FILTER"], gl_api["GL_LINEAR_MIPMAP_LINEAR"]
        )
        gl_api["glBindTexture"](target, 0)
        return texture

    miniworld_gl.Texture.load = _patched_load
    _TEXTURE_LOADER_PATCHED = True


def _ensure_extended_color_palette() -> None:
    global _COLOR_PALETTE_PATCHED
    if _COLOR_PALETTE_PATCHED:
        return

    try:
        import miniworld.entity as miniworld_entity  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("miniworld must be installed before applying patches.") from exc

    palette = {
        "white": np.array([1.0, 1.0, 1.0]),
        "orange": np.array([1.0, 0.55, 0.0]),
        "cyan": np.array([0.0, 1.0, 1.0]),
        "magenta": np.array([1.0, 0.0, 1.0]),
        "brown": np.array([0.59, 0.29, 0.0]),
        "pink": np.array([1.0, 0.75, 0.80]),
        "gray": np.array([0.39, 0.39, 0.39]),
    }

    existing = miniworld_entity.COLORS
    for name, value in palette.items():
        existing.setdefault(name, value)

    if "grey" in existing and "gray" not in existing:
        existing["gray"] = existing["grey"]
    if "gray" in existing and "grey" not in existing:
        existing["grey"] = existing["gray"]

    _COLOR_PALETTE_PATCHED = True


def _ensure_texture_fallbacks() -> None:
    global _TEXTURE_FALLBACK_PATCHED
    if _TEXTURE_FALLBACK_PATCHED:
        return

    try:
        import miniworld.opengl as miniworld_gl  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("miniworld must be installed before applying patches.") from exc

    from pyglet.image import SolidColorImagePattern

    original_get = miniworld_gl.Texture.get.__func__
    placeholder_cache: dict[str, miniworld_gl.Texture] = {}
    palette: dict[str, tuple[int, int, int, int]] = {
        "concrete": (180, 180, 180, 255),
        "brick": (160, 70, 50, 255),
        "wood": (160, 120, 80, 255),
        "grass": (80, 150, 80, 255),
        "carpet": (180, 100, 180, 255),
        "default": (200, 200, 200, 255),
    }

    def _placeholder(tex_name: str) -> miniworld_gl.Texture:
        cache_key = f"__fallback__/{tex_name}"
        cached = placeholder_cache.get(cache_key)
        if cached is not None:
            return cached
        base = tex_name.split("_", 1)[0].lower()
        color = palette.get(base, palette["default"])
        img = SolidColorImagePattern(color).create_image(4, 4)
        texture = img.get_texture()
        result = miniworld_gl.Texture(texture, tex_name)
        placeholder_cache[cache_key] = result
        return result

    @classmethod
    def _safe_get(cls, tex_name: str, rng=None):
        try:
            return original_get(cls, tex_name, rng)
        except AssertionError as exc:
            if "failed to load textures" not in str(exc):
                raise
            return _placeholder(tex_name)

    miniworld_gl.Texture.get = _safe_get
    _TEXTURE_FALLBACK_PATCHED = True


def _ensure_mesh_fallbacks() -> None:
    """Install a placeholder for missing mesh assets (e.g., building.obj)."""
    global _MESH_FALLBACK_PATCHED
    if _MESH_FALLBACK_PATCHED:
        return

    try:
        import miniworld.objmesh as miniworld_objmesh  # type: ignore
    except Exception:
        return

    original_objmesh_get = getattr(
        getattr(miniworld_objmesh.ObjMesh, "get", None), "__func__", None
    )
    if not callable(original_objmesh_get):
        return

    class DummyMesh:
        def __init__(self, mesh_name: str):
            self.max_coords = (1.0, 1.0, 1.0)
            self.min_coords = (0.0, 0.0, 0.0)
            self.mesh_name = mesh_name

        def render(self) -> None:
            return None

    @classmethod
    def safe_get(cls, mesh_name: str):
        try:
            return original_objmesh_get(cls, mesh_name)
        except Exception:
            return DummyMesh(mesh_name)

    miniworld_objmesh.ObjMesh.get = safe_get
    _MESH_FALLBACK_PATCHED = True


def _ensure_gl_legacy_symbols() -> None:
    global _GL_LEGACY_PATCHED
    if _GL_LEGACY_PATCHED:
        return

    try:
        import miniworld.miniworld as miniworld_core  # type: ignore
        import miniworld.opengl as miniworld_gl  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("miniworld must be installed before applying patches.") from exc

    modules = list(_get_gl_modules())
    modules.extend([miniworld_gl, miniworld_core])

    for name, value in LEGACY_GL_CONSTANTS.items():
        resolved = _lookup_gl_symbol(name) or value
        for module in modules:
            if not hasattr(module, name):
                setattr(module, name, resolved)

    for name in LEGACY_GL_FUNCTIONS:
        resolved = OPTIONAL_NOOPS.get(name) or _lookup_gl_symbol(name)
        if resolved is None:
            continue
        for module in modules:
            setattr(module, name, resolved)

    for name, value in LEGACY_GL_TYPES.items():
        resolved = _lookup_gl_symbol(name) or value
        for module in modules:
            if not hasattr(module, name):
                setattr(module, name, resolved)

    _GL_LEGACY_PATCHED = True
