"""Runtime 模块 - 统一的运行时接口"""

from typing import Union, Any, List


class Runtime:
    """统一运行时接口。

    支持从 CompiledProgram 对象创建，或从 .so 文件加载（预留）。
    """

    def __init__(self, compiled_or_path: Union[Any, str], device_id: int = 0):
        """初始化 Runtime

        Args:
            compiled_or_path: CompiledProgram 对象或 .so 文件路径
            device_id: CUDA 设备 ID
        """
        if isinstance(compiled_or_path, str):
            # 从 .so 文件加载（预留接口）
            from devproc.backend.triton import TritonCompiledProgram
            self._compiled = TritonCompiledProgram.load(compiled_or_path, device_id)
        else:
            self._compiled = compiled_or_path
        self._device_id = device_id

    def __call__(self, *args: Any, **kwargs: Any) -> List[Any]:
        """执行编译好的程序

        Args:
            *args: 位置输入参数
            **kwargs: 关键字输入参数

        Returns:
            输出张量列表
        """
        return self._compiled.run(*args, **kwargs)

    @property
    def device_id(self) -> int:
        """获取设备 ID"""
        return self._device_id
