from typing import List, Dict



def get_qa_pipeline(model_name: str = "deepset/roberta-base-squad2"):
    from transformers import pipeline  # lazy import to speed up app startup
    return pipeline("question-answering", model=model_name, tokenizer=model_name)


def answer_with_context(
    question: str,
    retrieved_chunks: List[Dict],
    qa_pipe,
    max_context_chars: int = 2000,
):
    texts: List[str] = []
    used_items: List[Dict] = []
    char_count = 0
    for item in retrieved_chunks:
        text = item["text"].strip()
        if not text:
            continue
        if char_count + len(text) > max_context_chars:
            break
        texts.append(text)
        used_items.append(item)
        char_count += len(text)

    context = "\n\n".join(texts)
    if not context:
        return {"answer": "No relevant context found in the PDF.", "score": 0.0}

    # Compute per-part offsets within the joined context for later highlighting
    parts = []
    offset = 0
    for idx, text in enumerate(texts):
        start_offset = offset
        end_offset = start_offset + len(text)
        parts.append({
            "index": idx,
            "page": int(used_items[idx].get("page", 0)),
            "start": start_offset,
            "end": end_offset,
            "text": text,
        })
        # Account for separator except after the last part
        offset = end_offset + (2 if idx < len(texts) - 1 else 0)

    result = qa_pipe(question=question, context=context)
    start_pos = int(result.get("start", -1)) if isinstance(result, dict) else -1
    end_pos = int(result.get("end", -1)) if isinstance(result, dict) else -1

    return {
        "answer": result.get("answer", ""),
        "score": float(result.get("score", 0.0)),
        "start": start_pos,
        "end": end_pos,
        "context": context,
        "parts": parts,
    }


