"""R-66: ``ChunkPreprocessor`` and ``ContextCompressor`` were byte-identical
ABCs, and implementations cross-inherited (``EmbeddingsChunkPreprocessor``, the
"preprocessor", subclasses ``ContextCompressor``). One interface now exists;
this name is kept as an alias so existing annotations keep resolving.
"""

from backend.domain.interfaces.context_compressor import ContextCompressor

ChunkPreprocessor = ContextCompressor

__all__ = ["ChunkPreprocessor"]
