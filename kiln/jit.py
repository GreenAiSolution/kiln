"""
kiln.jit - put machine code into memory and call it.

Apple Silicon enforces W^X: a page is never writable and executable at the
same moment. So we map it read/write, write the code, flip it to read/execute
with mprotect, and flush the instruction cache before the first call. That
last step matters - the CPU's I-cache does not snoop D-cache writes on ARM,
so skipping it means executing whatever was in that page before.

Pure standard library (ctypes + libc).
"""

import ctypes
import struct

_libc = ctypes.CDLL(None, use_errno=True)

_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.munmap.restype = ctypes.c_int
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mprotect.restype = ctypes.c_int
_libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.sys_icache_invalidate.restype = None
_libc.sys_icache_invalidate.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.posix_memalign.restype = ctypes.c_int
_libc.posix_memalign.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                 ctypes.c_size_t, ctypes.c_size_t]
_libc.free.restype = None
_libc.free.argtypes = [ctypes.c_void_p]

PROT_READ, PROT_WRITE, PROT_EXEC = 1, 2, 4
MAP_PRIVATE, MAP_ANON = 0x0002, 0x1000
PAGE = 16384          # Apple Silicon page size

_bytes_emitted = 0
_pages_mapped = 0


def stats():
    return {"code_bytes": _bytes_emitted, "pages": _pages_mapped}


class Code:
    """A page of executable machine code. Keep a reference or it unmaps."""

    __slots__ = ("addr", "size", "nbytes", "_alive")

    def __init__(self, blob):
        global _bytes_emitted, _pages_mapped
        if len(blob) == 0:
            raise ValueError("empty code")
        size = ((len(blob) + PAGE - 1) // PAGE) * PAGE
        p = _libc.mmap(None, size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANON, -1, 0)
        if p is None or p == (1 << 64) - 1:
            raise OSError(ctypes.get_errno(), "mmap failed")
        ctypes.memmove(p, blob, len(blob))
        if _libc.mprotect(ctypes.c_void_p(p), size, PROT_READ | PROT_EXEC) != 0:
            err = ctypes.get_errno()
            _libc.munmap(ctypes.c_void_p(p), size)
            raise OSError(err, "mprotect RX failed")
        _libc.sys_icache_invalidate(ctypes.c_void_p(p), size)
        self.addr = p
        self.size = size
        self.nbytes = len(blob)
        self._alive = True
        _bytes_emitted += len(blob)
        _pages_mapped += size // PAGE

    def fn(self, restype, argtypes):
        return ctypes.CFUNCTYPE(restype, *argtypes)(self.addr)

    def release(self):
        if self._alive:
            _libc.munmap(ctypes.c_void_p(self.addr), self.size)
            self._alive = False

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


VOIDP = ctypes.c_void_p
I64 = ctypes.c_int64
F32 = ctypes.c_float


def load(asm_or_bytes):
    blob = asm_or_bytes if isinstance(asm_or_bytes, (bytes, bytearray)) \
        else asm_or_bytes.code()
    return Code(bytes(blob))


class Buf:
    """A 64-byte-aligned float32 array. Aligned so vector loads never split a
    cache line, which otherwise costs ~15% on the wide kernels."""

    __slots__ = ("n", "_p", "ptr", "_arr")

    def __init__(self, n, fill=None):
        self.n = n
        raw = ctypes.c_void_p()
        nbytes = max(64, ((n * 4 + 63) // 64) * 64)
        if _libc.posix_memalign(ctypes.byref(raw), 64, nbytes) != 0:
            raise MemoryError("posix_memalign failed")
        self._p = raw
        self.ptr = raw.value
        self._arr = (ctypes.c_float * n).from_address(raw.value)
        if fill is not None:
            self.fill(fill)
        else:
            ctypes.memset(ctypes.c_void_p(self.ptr), 0, nbytes)

    def fill(self, v):
        ctypes.memset(ctypes.c_void_p(self.ptr), 0, self.n * 4)
        if v != 0.0:
            for i in range(self.n):
                self._arr[i] = v

    @classmethod
    def of(cls, seq):
        seq = list(seq)
        b = cls(len(seq))
        b[:] = seq
        return b

    def tolist(self):
        return list(self._arr)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self._arr[i]

    def __setitem__(self, i, v):
        if isinstance(i, slice):
            vals = list(v)
            idxs = range(*i.indices(self.n))
            if len(vals) != len(idxs):
                raise ValueError("length mismatch")
            for j, val in zip(idxs, vals):
                self._arr[j] = val
        else:
            self._arr[i] = v

    def frombytes(self, raw):
        if len(raw) != self.n * 4:
            raise ValueError("size mismatch")
        ctypes.memmove(ctypes.c_void_p(self.ptr), raw, len(raw))

    def tobytes(self):
        return ctypes.string_at(self.ptr, self.n * 4)

    def __del__(self):
        try:
            if self._p is not None:
                _libc.free(self._p)
                self._p = None
        except Exception:
            pass


def f32bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bitsf32(b):
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def f32(x):
    """Round a Python float to the nearest float32, as the hardware would."""
    return struct.unpack("<f", struct.pack("<f", x))[0]
