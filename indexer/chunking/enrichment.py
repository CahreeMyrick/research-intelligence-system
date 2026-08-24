from __future__ import annotations

from typing import Optional


def build_chunk_embedding_text(
    doc_title: Optional[str],
    section_path: list[str],
    chunk_source_text: str,
) -> str:
    """
    Constructs enriched embedding text for a chunk.
    Never mutates or overwrites source_text.
    
    Format:
        Document: <Title>
        Section: <Top-Level Section>
        Subsection: <Subsection>
        ...
        
        <Source Text>
    """
    lines: list[str] = []
    
    if doc_title:
        lines.append(f"Document: {doc_title.strip()}")
        
    if section_path:
        lines.append(f"Section: {section_path[0].strip()}")
        for sub in section_path[1:]:
            lines.append(f"Subsection: {sub.strip()}")
            
    header = "\n".join(lines)
    source = chunk_source_text.strip()
    
    if header and source:
        return f"{header}\n\n{source}"
    elif header:
        return header
    return source


def build_section_embedding_text(
    doc_title: Optional[str],
    section_path: list[str],
    section_body_text: str,
) -> str:
    """
    Constructs enriched embedding text for a whole section.
    
    Format:
        Document: <Title>
        Section: <Section Path>
        
        <Section Content>
    """
    lines: list[str] = []
    if doc_title:
        lines.append(f"Document: {doc_title.strip()}")
    if section_path:
        lines.append(f"Section: {' > '.join(s.strip() for s in section_path)}")
        
    header = "\n".join(lines)
    content = section_body_text.strip()
    
    if header and content:
        return f"{header}\n\n{content}"
    elif header:
        return header
    return content


def build_document_embedding_text(
    doc_title: Optional[str],
    abstract: Optional[str],
    section_titles: list[str],
) -> str:
    """
    Constructs high-level document embedding input.
    Focuses on Title, Abstract, and structural Outline of section titles.
    """
    parts: list[str] = []
    if doc_title:
        parts.append(f"Document: {doc_title.strip()}")
    if abstract:
        parts.append(f"Abstract:\n{abstract.strip()}")
    if section_titles:
        outline = "\n".join(f"- {t.strip()}" for t in section_titles if t.strip())
        parts.append(f"Sections:\n{outline}")
        
    return "\n\n".join(parts)
