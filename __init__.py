from .pandai_dsk_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__version__ = "2.1.0"
SUPPORT_PLATFORMS = ["openai", "volcengine"]
API_TIMEOUT = 30  # 统一超时时间

def validate_dependencies():
    import sys
    if sys.version_info < (3, 10):
        raise RuntimeError("火山引擎接口需要Python 3.10+版本")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
