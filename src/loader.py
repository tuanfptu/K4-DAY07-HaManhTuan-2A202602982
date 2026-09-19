from pathlib import Path

import yaml

from .models import Document


def load_markdown_documents(folder: str = "data/university") -> list[Document]:
    documents = []

    for path in sorted(Path(folder).glob("*.md")):
        text = path.read_text(encoding="utf-8")

        metadata = {}
        content = text

        # Parse YAML front matter
        if text.startswith("---"):
            parts = text.split("---", 2)

            if len(parts) == 3:
                metadata = yaml.safe_load(parts[1]) or {}
                content = parts[2].strip()

        doc = Document(
            id=path.stem,
            content=content,
            metadata=metadata,
        )

        documents.append(doc)

    return documents