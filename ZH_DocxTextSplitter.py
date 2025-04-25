import os
import re
import chardet
import zipfile
import xml.etree.ElementTree as ET
from docx import Document
import folder_paths
import traceback
import html

class ZH_DocxTextSplitter:
    def __init__(self):
        self.debug_mode = True

    def debug_log(self, message):
        if self.debug_mode:
            print(f"[ZH_DEBUG] {message}")

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        try:
            files = sorted(
                [f for f in os.listdir(input_dir) 
                if f.lower().endswith((".docx", ".doc", ".txt"))],
                key=lambda x: x.lower()
            )
        except Exception as e:
            print(f"文件列表获取失败: {str(e)}")
        
        extra_paths = []
        for folder in folder_paths.folder_names_and_paths:
            if isinstance(folder_paths.folder_names_and_paths[folder], list):
                for path in folder_paths.folder_names_and_paths[folder]:
                    extra_paths.append(f"{folder}: {os.path.basename(path)}")

        return {
            "required": {
                "文档文件": (["None"] + files,),
                "分隔符": ("STRING", {"default": "\n", "dynamicPrompts": False}),
                "起始位置": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "跳过间隔": ("INT", {"default": 0, "min": 0, "max": 10}),
                "最大数量": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "编码检测": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "自定义路径": ("STRING", {"default": "", "multiline": False}),
                "输出编码": (["utf-8", "gbk", "auto"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本段落",)
    FUNCTION = "process"
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "ZH-处理工具"

    def process(self, 文档文件, 分隔符, 起始位置, 跳过间隔, 最大数量, 编码检测, 自定义路径="", 输出编码="auto"):
        try:
            # 文件路径处理
            file_path = self.get_file_path(文档文件, 自定义路径)
            self.debug_log(f"最终文件路径: {file_path}")

            # 读取文件内容
            content = self.read_file(file_path, 编码检测)
            if not content:
                return (["错误：无法读取文件内容"],)

            # 分割文本
            segments = self.split_content(content, 分隔符)
            self.debug_log(f"初步分割段落数: {len(segments)}")

            # 应用处理参数
            final_segments = segments[起始位置 : 起始位置 + 最大数量*(跳过间隔+1) : 跳过间隔+1]
            self.debug_log(f"最终输出段落数: {len(final_segments)}")

            # 编码转换
            if 输出编码 != "auto":
                final_segments = [s.encode(输出编码, errors='ignore').decode(输出编码) for s in final_segments]

            return (final_segments,)
        except Exception as e:
            error_msg = f"处理失败: {str(e)}\n{traceback.format_exc()}"
            self.debug_log(error_msg)
            return ([error_msg],)

    def get_file_path(self, selected_file, custom_path):
        if custom_path and os.path.exists(custom_path):
            return custom_path
            
        input_dir = folder_paths.get_input_directory()
        default_path = os.path.join(input_dir, selected_file)
        
        if os.path.exists(default_path):
            return default_path
            
        # 尝试修复路径
        for ext in ['.txt', '.docx', '.doc']:
            test_path = default_path + ext
            if os.path.exists(test_path):
                return test_path
                
        raise FileNotFoundError(f"文件不存在: {selected_file}")

    def read_file(self, file_path, detect_encoding):
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt':
            return self.read_text_file(file_path, detect_encoding)
        elif ext in ('.docx', '.doc'):
            return self.read_docx(file_path)
        else:
            raise ValueError("不支持的文件格式")

    def read_text_file(self, file_path, detect_encoding):
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            
        encoding = chardet.detect(raw_data)['encoding'] if detect_encoding else 'utf-8'
        try:
            return raw_data.decode(encoding, errors='replace')
        except:
            return raw_data.decode('utf-8', errors='replace')

    def read_docx(self, file_path):
        try:
            doc = Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
        except:
            return self.read_docx_fallback(file_path)

    def read_docx_fallback(self, file_path):
        try:
            with zipfile.ZipFile(file_path) as z:
                xml_content = z.read('word/document.xml')
                root = ET.fromstring(xml_content)
                return '\n'.join(
                    [node.text for node in root.iter() if node.text and node.text.strip()]
                )
        except Exception as e:
            raise ValueError(f"DOCX解析失败: {str(e)}")

    def split_content(self, content, delimiter):
        try:
            delimiter = delimiter.encode().decode('unicode_escape')
        except:
            pass
            
        if not delimiter:
            return [content]
            
        return [s.strip() for s in content.split(delimiter) if s.strip()]

NODE_CLASS_MAPPINGS = {"ZH_DocxTextSplitter": ZH_DocxTextSplitter}
NODE_DISPLAY_NAME_MAPPINGS = {"ZH_DocxTextSplitter": "ZH-文档处理器"}
