from pathlib import Path

from finvoice_ai.application.ports import RetrievedDocument


class InvalidDocumentError(ValueError):
    """Raised when an approved knowledge document does not match the contract."""


class MarkdownDocumentLoader:
    """Load repository-controlled Markdown files into retrieval documents."""

    def load_directory(self, directory: Path) -> list[RetrievedDocument]:
        documents = [self.load(path) for path in sorted(directory.glob("*.md"))]
        if not documents:
            raise InvalidDocumentError(f"no Markdown documents found in {directory}")
        return documents

    def load(self, path: Path) -> RetrievedDocument:
        text = path.read_text(encoding="utf-8").strip()
        lines = text.splitlines()
        if not lines or not lines[0].startswith("# "):
            raise InvalidDocumentError(f"{path.name} must start with a level-one heading")

        title = lines[0].removeprefix("# ").strip()
        content = " ".join(line.strip() for line in lines[1:] if line.strip())
        if not title or not content:
            raise InvalidDocumentError(f"{path.name} must contain a title and body")

        return RetrievedDocument(
            document_id=path.stem,
            title=title,
            content=content,
            source=f"knowledge/{path.name}",
        )


def load_default_documents() -> list[RetrievedDocument]:
    knowledge_directory = Path(__file__).parents[1] / "knowledge"
    return MarkdownDocumentLoader().load_directory(knowledge_directory)
