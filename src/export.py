from typing import Literal, Any
from pathlib import Path
from src.schemas import RagAnswer, Summary, QuizSet, FlashcardSet

ExportFormat = Literal["text", "md", "json"]

def _to_markdown(model: Any) -> str:
    if isinstance(model, RagAnswer):
        md = [f"# Hỏi đáp RAG\n", f"**Câu hỏi:** {model.question}\n", f"**Trả lời:**\n\n{model.answer}\n"]
        if model.citations:
            md.append("## Nguồn trích dẫn:\n")
            for c in model.citations:
                md.append(f"- **{c.source_marker}**: {c.filename} (Trang {c.page})\n")
        return "\n".join(md)
        
    elif isinstance(model, Summary):
        scope_str = f"Tài liệu {model.target}" if model.scope == "document" else "Toàn bộ tài liệu"
        md = [
            f"# Tóm tắt: {scope_str}\n",
            f"## Bản tóm tắt chính:\n\n{model.summary}\n",
            "## Các ý chính nổi bật:\n"
        ]
        for kp in model.key_points:
            md.append(f"- {kp}\n")
        if model.citations:
            md.append("\n## Nguồn tham chiếu:\n")
            for c in model.citations:
                md.append(f"- **{c.source_marker}**: {c.filename} (Trang {c.page})\n")
        return "\n".join(md)
        
    elif isinstance(model, QuizSet):
        md = [f"# Trắc nghiệm ôn tập\n"]
        for i, item in enumerate(model.items, start=1):
            md.append(f"### Câu {i}: {item.question}\n")
            options_labels = ["A", "B", "C", "D"]
            for opt_idx, opt in enumerate(item.options):
                marker = " [x]" if opt_idx == item.correct_index else " [ ]"
                md.append(f"-{marker} {options_labels[opt_idx]}. {opt}\n")
            md.append(f"\n*Giải thích:* {item.explanation}\n")
            if item.source_markers:
                md.append(f"*Tham chiếu:* {', '.join(item.source_markers)}\n")
            md.append("\n" + "-"*40 + "\n")
        return "\n".join(md)
        
    elif isinstance(model, FlashcardSet):
        md = [f"# Thẻ ghi nhớ (Flashcards)\n"]
        for i, card in enumerate(model.cards, start=1):
            md.append(f"### Flashcard {i}:\n")
            md.append(f"**Mặt trước (Front):** {card.front}\n")
            md.append(f"**Mặt sau (Back):** {card.back}\n")
            if card.hint:
                md.append(f"**Gợi ý (Hint):** {card.hint}\n")
            if card.topic:
                md.append(f"**Chủ đề (Topic):** {card.topic}\n")
            if card.source_markers:
                md.append(f"**Tham chiếu:** {', '.join(card.source_markers)}\n")
            md.append("\n" + "-"*40 + "\n")
        return "\n".join(md)
        
    return str(model)

def export(model: Any, *, fmt: ExportFormat = "text", output: str = None) -> Any:
    if fmt == "json":
        text = model.model_dump_json(indent=2) + "\n"
    elif fmt in {"text", "md"}:
        text = _to_markdown(model)
    else:
        raise ValueError(f"Unknown fmt '{fmt}'. Expected 'text' | 'md' | 'json'.")
        
    if output is None:
        return text
        
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path
