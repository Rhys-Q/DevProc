"""
CUDA Driver API Wrapper.

Provides low-level access to CUDA driver functions via ctypes.
Used for loading and executing AOT-compiled kernels.
"""

import ctypes
from ctypes import c_void_p, c_int, c_uint, c_size_t, c_char_p, POINTER, byref
from typing import Tuple, Optional, Any, List
import logging

logger = logging.getLogger(__name__)


class CUDARuntimeError(Exception):
    """CUDA runtime error."""
    pass


class CUDADriver:
    """CUDA Driver API wrapper using ctypes."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._lib = None
        self._device: Optional[c_void_p] = None
        self._context: Optional[c_void_p] = None
        self._initialized = True
        self._modules: List[c_void_p] = []

    def _try_load_library(self) -> bool:
        """Try to load CUDA driver library."""
        possible_libs = [
            "libcuda.so.1",
            "libcuda.so",
            "/usr/local/cuda/lib64/libcuda.so.1",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        ]

        for lib_name in possible_libs:
            try:
                self._lib = ctypes.CDLL(lib_name)
                logger.info(f"Loaded CUDA driver: {lib_name}")
                return True
            except OSError:
                continue

        logger.warning("Could not load CUDA driver library")
        return False

    @property
    def is_available(self) -> bool:
        """Check if CUDA driver is available."""
        if self._lib is None:
            return self._try_load_library()
        return True

    def _check_error(self, result: c_int, func_name: str) -> None:
        """Check CUDA API return value for errors."""
        if result != 0:
            raise CUDARuntimeError(
                f"CUDA API error in {func_name}: error code {result}"
            )

    def init(self) -> None:
        """Initialize CUDA driver."""
        if self._lib is None:
            if not self._try_load_library():
                raise CUDARuntimeError("CUDA driver not available")

        # Setup function prototypes
        self._setup_functions()

        # Initialize CUDA
        self._check_error(self._lib.cuInit(0), "cuInit")

    def _setup_functions(self) -> None:
        """Setup CUDA driver function prototypes."""
        lib = self._lib

        # cuDeviceGet
        lib.cuDeviceGet.argtypes = [POINTER(c_void_p), c_int]
        lib.cuDeviceGet.restype = c_int

        # cuCtxCreate_v2
        lib.cuCtxCreate_v2.argtypes = [POINTER(c_void_p), c_uint, c_void_p]
        lib.cuCtxCreate_v2.restype = c_int

        # cuCtxSetCurrent
        lib.cuCtxSetCurrent.argtypes = [c_void_p]
        lib.cuCtxSetCurrent.restype = c_int

        # cuCtxGetCurrent
        lib.cuCtxGetCurrent.argtypes = [POINTER(c_void_p)]
        lib.cuCtxGetCurrent.restype = c_int

        # cuModuleLoad
        lib.cuModuleLoad.argtypes = [POINTER(c_void_p), c_char_p]
        lib.cuModuleLoad.restype = c_int

        # cuModuleLoadData
        lib.cuModuleLoadData.argtypes = [POINTER(c_void_p), c_void_p]
        lib.cuModuleLoadData.restype = c_int

        # cuModuleUnload
        lib.cuModuleUnload.argtypes = [c_void_p]
        lib.cuModuleUnload.restype = c_int

        # cuModuleGetFunction
        lib.cuModuleGetFunction.argtypes = [
            POINTER(c_void_p),
            c_void_p,
            c_char_p,
        ]
        lib.cuModuleGetFunction.restype = c_int

        # cuMemAlloc_v2
        lib.cuMemAlloc_v2.argtypes = [POINTER(c_void_p), c_size_t]
        lib.cuMemAlloc_v2.restype = c_int

        # cuMemFree_v2
        lib.cuMemFree_v2.argtypes = [c_void_p]
        lib.cuMemFree_v2.restype = c_int

        # cuMemcpyHtoD_v2
        lib.cuMemcpyHtoD_v2.argtypes = [c_void_p, c_void_p, c_size_t]
        lib.cuMemcpyHtoD_v2.restype = c_int

        # cuMemcpyDtoH_v2
        lib.cuMemcpyDtoH_v2.argtypes = [c_void_p, c_void_p, c_size_t]
        lib.cuMemcpyDtoH_v2.restype = c_int

        # cuLaunchKernel
        lib.cuLaunchKernel.argtypes = [
            c_void_p,  # function
            c_uint, c_uint, c_uint,  # grid x, y, z
            c_uint, c_uint, c_uint,  # block x, y, z
            c_uint,  # shared memory
            c_void_p,  # stream
            POINTER(c_void_p),  # args
            POINTER(c_void_p),  # extra
        ]
        lib.cuLaunchKernel.restype = c_int

        # cuLaunchKernel is also known as cuLaunchKernel_v2
        try:
            lib.cuLaunchKernel_v2 = lib.cuLaunchKernel
        except AttributeError:
            pass

        # cuDeviceGetAttribute (for getting device properties)
        lib.cuDeviceGetAttribute.argtypes = [POINTER(c_int), c_int, c_void_p]
        lib.cuDeviceGetAttribute.restype = c_int

    def get_device(self, device_id: int = 0) -> c_void_p:
        """Get CUDA device handle."""
        device = c_void_p()
        self._check_error(
            self._lib.cuDeviceGet(byref(device), device_id),
            "cuDeviceGet"
        )
        self._device = device
        return device

    def create_context(self, device: Optional[c_void_p] = None) -> c_void_p:
        """Create CUDA context."""
        if device is None:
            device = self.get_device()

        context = c_void_p()
        self._check_error(
            self._lib.cuCtxCreate_v2(byref(context), 0, device),
            "cuCtxCreate_v2"
        )
        self._context = context
        return context

    def set_context(self, context: c_void_p) -> None:
        """Set current CUDA context."""
        self._check_error(
            self._lib.cuCtxSetCurrent(context),
            "cuCtxSetCurrent"
        )
        self._context = context

    def get_context(self) -> Optional[c_void_p]:
        """Get current CUDA context."""
        context = c_void_p()
        self._lib.cuCtxGetCurrent(byref(context))
        return context

    def load_module(self, path: str) -> c_void_p:
        """Load a CUDA module from file."""
        module = c_void_p()
        self._check_error(
            self._lib.cuModuleLoad(byref(module), path.encode()),
            "cuModuleLoad"
        )
        self._modules.append(module)
        return module

    def load_module_from_memory(self, cubin: bytes) -> c_void_p:
        """Load a CUDA module from memory (cubin binary)."""
        module = c_void_p()

        # Create a mutable buffer from the cubin bytes
        cubin_buffer = ctypes.create_string_buffer(cubin)

        self._check_error(
            self._lib.cuModuleLoadData(byref(module), cubin_buffer),
            "cuModuleLoadData"
        )
        self._modules.append(module)
        return module

    def unload_module(self, module: c_void_p) -> None:
        """Unload a CUDA module."""
        self._check_error(self._lib.cuModuleUnload(module), "cuModuleUnload")

    def get_function(self, module: c_void_p, name: str) -> c_void_p:
        """Get a kernel function from a module."""
        function = c_void_p()
        self._check_error(
            self._lib.cuModuleGetFunction(byref(function), module, name.encode()),
            "cuModuleGetFunction"
        )
        return function

    def allocate_memory(self, size: int) -> c_void_p:
        """Allocate GPU memory."""
        ptr = c_void_p()
        self._check_error(
            self._lib.cuMemAlloc_v2(byref(ptr), size),
            "cuMemAlloc_v2"
        )
        return ptr

    def free_memory(self, ptr: c_void_p) -> None:
        """Free GPU memory."""
        self._check_error(self._lib.cuMemFree_v2(ptr), "cuMemFree_v2")

    def copy_host_to_device(self, dst: c_void_p, src: Any, size: int) -> None:
        """Copy data from host to device."""
        # Get the data pointer from the source
        if hasattr(src, 'data_ptr'):
            src_ptr = src.data_ptr()
        else:
            src_ptr = ctypes.addressof(src)

        self._check_error(
            self._lib.cuMemcpyHtoD_v2(dst, src_ptr, size),
            "cuMemcpyHtoD_v2"
        )

    def copy_device_to_host(self, dst: Any, src: c_void_p, size: int) -> None:
        """Copy data from device to host."""
        # Get the data pointer from the destination
        if hasattr(dst, 'data_ptr'):
            dst_ptr = dst.data_ptr()
        else:
            dst_ptr = ctypes.addressof(dst)

        self._check_error(
            self._lib.cuMemcpyDtoH_v2(dst_ptr, src, size),
            "cuMemcpyDtoH_v2"
        )

    def launch_kernel(
        self,
        function: c_void_p,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        shared_memory: int = 0,
        args: Optional[List[c_void_p]] = None,
        stream: Optional[c_void_p] = None,
    ) -> None:
        """Launch a CUDA kernel."""
        # Prepare arguments
        if args is None:
            args = []

        # Create array of pointers
        args_array = (c_void_p * len(args))(*args)
        extra = None

        self._check_error(
            self._lib.cuLaunchKernel(
                function,
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                shared_memory,
                stream if stream else c_void_p(),
                args_array,
                extra,
            ),
            "cuLaunchKernel"
        )

    def get_device_attribute(self, attr: int, device: Optional[c_void_p] = None) -> int:
        """Get device attribute."""
        if device is None:
            device = self.get_device()

        value = c_int()
        self._check_error(
            self._lib.cuDeviceGetAttribute(byref(value), attr, device),
            "cuDeviceGetAttribute"
        )
        return value.value


# Global driver instance
_driver: Optional[CUDADriver] = None


def get_driver() -> CUDADriver:
    """Get global CUDA driver instance."""
    global _driver
    if _driver is None:
        _driver = CUDADriver()
    return _driver


def init_cuda() -> CUDADriver:
    """Initialize CUDA driver."""
    driver = get_driver()
    driver.init()
    return driver
