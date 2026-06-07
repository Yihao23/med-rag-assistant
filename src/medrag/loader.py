"""从磁盘加载文档(支持 .txt / .md / .pdf)。"""

from __future__ import annotations

from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def load_document(path: Path) -> str:
    """读取单个文件,返回纯文本。"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"不支持的文件类型:{suffix}")


def load_directory(directory: Path) -> dict[str, str]:
    """加载目录下所有支持的文档,返回 {文件名: 文本}。"""
    docs: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            docs[path.name] = load_document(path)
    return docs
