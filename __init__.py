from .pandai_dsk_node import NODE_CLASS_MAPPINGS as DSK_NODES
from .ZH_DocxTextSplitter import NODE_CLASS_MAPPINGS as DOCX_NODES

NODE_CLASS_MAPPINGS = {**DSK_NODES, **DOCX_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {
    "pandai_dsk_node": "Pandai DSK Node",
    "ZH_DocxTextSplitter": "ZH-Docx文本分割器"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
