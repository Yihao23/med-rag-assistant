import pytest

from medrag.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("   ", source="a.txt") == []


def test_chunks_cover_whole_text_with_overlap():
    text = "abcdefghij" * 20  # 200 字符
    chunks = chunk_text(text, source="a.txt", size=80, overlap=20)
    assert len(chunks) >= 2
    assert all(c.source == "a.txt" for c in chunks)
    assert all(len(c.text) <= 80 for c in chunks)
    # 重叠:第二块的开头应出现在第一块里
    assert chunks[1].text[:10] in chunks[0].text


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("hello", source="a.txt", size=10, overlap=10)
