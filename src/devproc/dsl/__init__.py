# DevProc DSL Module
"""
DSL layer for building DevProc IR pipelines.

Usage:
    from devproc import pipeline

    pipe = pipeline()
    img = pipe.input("image", "uint8", (224, 224, 3))
    x = pipe.normalize(img)
    out = pipe.argmax(x)
    pipe.output(out)

    # Build IR
    ir_func = pipe.build()
"""

from devproc.dsl.pipeline import Pipeline, Tensor

__all__ = [
    "Pipeline",
    "Tensor",
]
