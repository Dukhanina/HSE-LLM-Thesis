import os
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MODE_ORDER = ["template", "model_without_references", "model_with_references"]
MODE_LABELS = {
    "template": "Шаблонный способ",
    "model_without_references": "Удаленная модель без аналогов",
    "model_with_references": "Удаленная модель с аналогами",
}

DOCUMENT_SECTIONS = [
    "Название проекта",
    "Краткая концепция",
    "Жанр и целевая аудитория",
    "Основной игровой цикл",
    "Ключевые механики",
    "Платформа",
    "Художественный ориентир",
    "Ограничение по масштабу",
    "Игровые аналоги",
    "Основные риски",
    "Первый план работ",
]

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key")

PROXYAPI_API_KEY = os.getenv("PROXYAPI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openai.api.proxyapi.ru/v1").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini").strip()
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))

client: Optional[OpenAI] = None
if PROXYAPI_API_KEY:
    client = OpenAI(
        api_key=PROXYAPI_API_KEY,
        base_url=OPENAI_BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def to_plain_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def row_to_dict(row: pd.Series) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in row.items():
        result[str(key)] = to_plain_value(value)
    return result


def require_file(filename: str) -> Path:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл данных: {path}")
    return path


def load_dataframes() -> Dict[str, pd.DataFrame]:
    briefs = pd.read_csv(require_file("test_briefs_stage_4.csv"))
    references = pd.read_csv(require_file("references_unique_stage_6.csv"))
    generated_docs = pd.read_csv(require_file("generated_documents_stage_6.csv"))
    metrics = pd.read_csv(require_file("document_metrics_stage_7.csv"))
    comparison = pd.read_csv(require_file("comparison_table_stage_9.csv"))
    final_mode = pd.read_csv(require_file("final_mode_table_stage_10.csv"))

    # Для корпуса игр загружаем только действительно нужные поля
    corpus_needed_columns = {
        "id",
        "name",
        "platform_family",
        "anchor_genre",
        "genre_names_text",
        "text_description",
        "description_length",
        "release_year_valid",
    }
    corpus = pd.read_csv(
        require_file("study_corpus_stage_4.csv.gz"),
        compression="gzip",
        usecols=lambda c: c in corpus_needed_columns,
    )

    # Базовая нормализация
    for df in [briefs, references, generated_docs, metrics, comparison, final_mode, corpus]:
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("")

    return {
        "briefs": briefs,
        "references": references,
        "generated_docs": generated_docs,
        "metrics": metrics,
        "comparison": comparison,
        "final_mode": final_mode,
        "corpus": corpus,
    }


DATA = load_dataframes()

BRIEFS_DF = DATA["briefs"].copy()
REFERENCES_DF = DATA["references"].copy()
GENERATED_DOCS_DF = DATA["generated_docs"].copy()
METRICS_DF = DATA["metrics"].copy()
COMPARISON_DF = DATA["comparison"].copy()
FINAL_MODE_DF = DATA["final_mode"].copy()
CORPUS_DF = DATA["corpus"].copy()

BRIEFS_BY_ID: Dict[str, Dict[str, Any]] = {
    str(row["task_id"]): row_to_dict(row)
    for _, row in BRIEFS_DF.iterrows()
}

TASK_IDS: List[str] = list(BRIEFS_BY_ID.keys())


def get_filters() -> Dict[str, List[str]]:
    genres = sorted(
        [x for x in BRIEFS_DF["anchor_genre"].dropna().astype(str).unique().tolist() if x.strip()]
    )
    platforms = sorted(
        [x for x in BRIEFS_DF["platform_family"].dropna().astype(str).unique().tolist() if x.strip()]
    )
    return {"genres": genres, "platforms": platforms}


def build_task_list(genre: str = "", platform: str = "", query: str = "") -> List[Dict[str, Any]]:
    df = BRIEFS_DF.copy()

    if genre:
        df = df[df["anchor_genre"].astype(str) == genre]

    if platform:
        df = df[df["platform_family"].astype(str) == platform]

    if query:
        q = query.strip().lower()
        mask = (
            df["task_id"].astype(str).str.lower().str.contains(q, na=False)
            | df["source_game_name"].astype(str).str.lower().str.contains(q, na=False)
            | df["brief_text"].astype(str).str.lower().str.contains(q, na=False)
        )
        df = df[mask]

    df = df.sort_values(["anchor_genre", "platform_family", "source_game_name"]).reset_index(drop=True)

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        records.append(
            {
                "task_id": safe_text(row["task_id"]),
                "source_game_name": safe_text(row["source_game_name"]),
                "anchor_genre": safe_text(row["anchor_genre"]),
                "platform_family": safe_text(row["platform_family"]),
                "release_year_valid": to_plain_value(row.get("release_year_valid")),
            }
        )
    return records


def build_reference_packet(task_id: str, limit: int = 4) -> str:
    part = REFERENCES_DF[REFERENCES_DF["task_id"].astype(str) == str(task_id)].copy()
    part = part.sort_values(["reference_rank", "similarity_score"], ascending=[True, False]).head(limit)

    if part.empty:
        return "Подходящие игровые аналоги не найдены."

    lines: List[str] = []
    for idx, (_, row) in enumerate(part.iterrows(), start=1):
        description = safe_text(row.get("reference_text_description", ""))
        if len(description) > 180:
            description = description[:180].rsplit(" ", 1)[0] + "..."

        lines.append(
            f"Аналог {idx}. "
            f"Название: {safe_text(row.get('reference_game_name', ''))}. "
            f"Семейство платформ: {safe_text(row.get('reference_platform_family', ''))}. "
            f"Главный жанр: {safe_text(row.get('reference_anchor_genre', ''))}. "
            f"Полный перечень жанров: {safe_text(row.get('reference_genres_text', ''))}. "
            f"Год выхода: {safe_text(row.get('reference_release_year', ''))}. "
            f"Краткое описание: {description}"
        )
    return "\n".join(lines)


def build_system_prompt() -> str:
    section_list_text = "\n".join([f"- {section_name}" for section_name in DOCUMENT_SECTIONS])
    return (
        "Необходимо подготовить краткий проектный документ игры на русском языке. "
        "Документ должен быть конкретным, сдержанным и пригодным для учебной проектной работы высокого уровня. "
        "Запрещено ссылаться на то, что текст был создан моделью. "
        "Запрещено использовать таблицы. "
        "Необходимо строго соблюдать следующую структуру и порядок разделов:\n"
        f"{section_list_text}\n"
        "Каждый раздел должен начинаться с заголовка вида ### Название раздела. "
        "После каждого заголовка требуется 2 или 3 предложения содержательного текста. "
        "Нужно предложить новое рабочее название проекта, не повторяя название исходной игры и не копируя названия аналогов. "
        "Текст должен быть связным, предметным и не шаблонно-общим."
    )


def build_generation_prompt(task: Dict[str, Any], use_references: bool, extra_instruction: str) -> str:
    base_prompt = (
        "Требуется подготовить краткий проектный документ игры.\n\n"
        "Текст задания:\n"
        f"{safe_text(task.get('brief_text', ''))}\n\n"
    )

    if use_references:
        base_prompt += (
            "Найденные игровые аналоги:\n"
            f"{build_reference_packet(safe_text(task.get('task_id', '')))}\n\n"
            "Необходимо использовать аналоги как ориентир, но не копировать их напрямую. "
            "Следует сохранить самостоятельность проектного замысла.\n\n"
        )
    else:
        base_prompt += "Игровые аналоги использовать нельзя. Необходимо опираться только на сведения из задания.\n\n"

    if extra_instruction.strip():
        base_prompt += f"Дополнительное указание пользователя:\n{extra_instruction.strip()}\n"

    return base_prompt


def generate_live_document(task_id: str, use_references: bool, extra_instruction: str) -> Dict[str, Any]:
    if client is None:
        raise RuntimeError("Ключ Proxy API не найден в .env.")

    task = BRIEFS_BY_ID.get(task_id)
    if not task:
        raise ValueError("Задание не найдено.")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {
                "role": "user",
                "content": build_generation_prompt(
                    task=task,
                    use_references=use_references,
                    extra_instruction=extra_instruction,
                ),
            },
        ],
        temperature=0.35,
        max_tokens=950,
    )

    text = safe_text(response.choices[0].message.content)
    if not text:
        raise RuntimeError("Модель вернула пустой ответ.")

    return {
        "task_id": task_id,
        "model_name": MODEL_NAME,
        "use_references": use_references,
        "generated_text": text,
    }


def get_task_payload(task_id: str) -> Dict[str, Any]:
    task = BRIEFS_BY_ID.get(task_id)
    if not task:
        raise KeyError("Задание не найдено.")

    refs = REFERENCES_DF[REFERENCES_DF["task_id"].astype(str) == task_id].copy()
    refs = refs.sort_values(["reference_rank", "similarity_score"], ascending=[True, False])

    ref_records: List[Dict[str, Any]] = []
    for _, row in refs.iterrows():
        ref_records.append(
            {
                "reference_rank": to_plain_value(row.get("reference_rank")),
                "reference_game_name": safe_text(row.get("reference_game_name")),
                "reference_platform_family": safe_text(row.get("reference_platform_family")),
                "reference_anchor_genre": safe_text(row.get("reference_anchor_genre")),
                "reference_genres_text": safe_text(row.get("reference_genres_text")),
                "reference_release_year": to_plain_value(row.get("reference_release_year")),
                "similarity_score": round(float(row.get("similarity_score", 0.0)), 4)
                if safe_text(row.get("similarity_score", "")) != ""
                else None,
                "reference_text_description": safe_text(row.get("reference_text_description")),
            }
        )

    docs = GENERATED_DOCS_DF[GENERATED_DOCS_DF["task_id"].astype(str) == task_id].copy()
    metrics = METRICS_DF[METRICS_DF["task_id"].astype(str) == task_id].copy()

    doc_records: List[Dict[str, Any]] = []
    for mode in MODE_ORDER:
        doc_row = docs[docs["mode"].astype(str) == mode]
        metric_row = metrics[metrics["mode"].astype(str) == mode]

        doc_data = doc_row.iloc[0] if not doc_row.empty else None
        metric_data = metric_row.iloc[0] if not metric_row.empty else None

        doc_records.append(
            {
                "mode": mode,
                "mode_label": MODE_LABELS.get(mode, mode),
                "generated_text": safe_text(doc_data.get("generated_text")) if doc_data is not None else "",
                "text_length_chars": to_plain_value(doc_data.get("text_length_chars")) if doc_data is not None else None,
                "section_count": to_plain_value(doc_data.get("section_count")) if doc_data is not None else None,
                "reference_count": to_plain_value(doc_data.get("reference_count")) if doc_data is not None else None,
                "expert_total_score": to_plain_value(metric_data.get("expert_total_score")) if metric_data is not None else None,
                "task_fit": to_plain_value(metric_data.get("task_fit")) if metric_data is not None else None,
                "genre_fit": to_plain_value(metric_data.get("genre_fit")) if metric_data is not None else None,
                "mechanics_specificity": to_plain_value(metric_data.get("mechanics_specificity")) if metric_data is not None else None,
                "scope_realism": to_plain_value(metric_data.get("scope_realism")) if metric_data is not None else None,
                "risk_quality": to_plain_value(metric_data.get("risk_quality")) if metric_data is not None else None,
                "coherence": to_plain_value(metric_data.get("coherence")) if metric_data is not None else None,
            }
        )

    comparison_row = COMPARISON_DF[COMPARISON_DF["task_id"].astype(str) == task_id]
    comparison_data = row_to_dict(comparison_row.iloc[0]) if not comparison_row.empty else {}

    return {
        "task": task,
        "references": ref_records,
        "documents": doc_records,
        "comparison": comparison_data,
    }


def build_analytics_payload() -> Dict[str, Any]:
    comparison = COMPARISON_DF.copy()

    reference_effect_summary = (
        comparison.groupby("reference_effect_group", as_index=False)
        .agg(tasks_count=("task_id", "count"))
        .sort_values("tasks_count", ascending=False)
        .reset_index(drop=True)
    )

    genre_delta_summary = (
        comparison.groupby("anchor_genre", as_index=False)
        .agg(
            mean_delta_model_vs_template=("delta_model_vs_template", "mean"),
            mean_delta_model_refs_vs_template=("delta_model_refs_vs_template", "mean"),
            mean_delta_refs_vs_no_refs=("delta_refs_vs_no_refs", "mean"),
        )
        .sort_values("mean_delta_refs_vs_no_refs", ascending=False)
        .reset_index(drop=True)
    )

    platform_delta_summary = (
        comparison.groupby("platform_family", as_index=False)
        .agg(
            mean_delta_model_vs_template=("delta_model_vs_template", "mean"),
            mean_delta_model_refs_vs_template=("delta_model_refs_vs_template", "mean"),
            mean_delta_refs_vs_no_refs=("delta_refs_vs_no_refs", "mean"),
        )
        .sort_values("mean_delta_refs_vs_no_refs", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "final_mode_table": [row_to_dict(row) for _, row in FINAL_MODE_DF.iterrows()],
        "reference_effect_summary": [row_to_dict(row) for _, row in reference_effect_summary.iterrows()],
        "genre_delta_summary": [row_to_dict(row) for _, row in genre_delta_summary.iterrows()],
        "platform_delta_summary": [row_to_dict(row) for _, row in platform_delta_summary.iterrows()],
    }


def search_corpus(query: str, genre: str = "", platform: str = "", limit: int = 20) -> List[Dict[str, Any]]:
    df = CORPUS_DF.copy()

    if genre:
        df = df[df["anchor_genre"].astype(str) == genre]
    if platform:
        df = df[df["platform_family"].astype(str) == platform]

    query = query.strip().lower()
    if query:
        mask = (
            df["name"].astype(str).str.lower().str.contains(query, na=False)
            | df["genre_names_text"].astype(str).str.lower().str.contains(query, na=False)
            | df["text_description"].astype(str).str.lower().str.contains(query, na=False)
        )
        df = df[mask]

    df = df.head(limit)

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        description = safe_text(row.get("text_description", ""))
        if len(description) > 220:
            description = description[:220].rsplit(" ", 1)[0] + "..."

        records.append(
            {
                "id": to_plain_value(row.get("id")),
                "name": safe_text(row.get("name")),
                "platform_family": safe_text(row.get("platform_family")),
                "anchor_genre": safe_text(row.get("anchor_genre")),
                "genre_names_text": safe_text(row.get("genre_names_text")),
                "release_year_valid": to_plain_value(row.get("release_year_valid")),
                "description_length": to_plain_value(row.get("description_length")),
                "text_description": description,
            }
        )
    return records


@app.route("/")
def index():
    filters = get_filters()
    default_task_id = TASK_IDS[0] if TASK_IDS else ""
    return render_template(
        "index.html",
        filters=filters,
        default_task_id=default_task_id,
        model_name=MODEL_NAME,
        model_enabled=bool(client),
    )


@app.route("/api/tasks")
def api_tasks():
    genre = request.args.get("genre", "").strip()
    platform = request.args.get("platform", "").strip()
    query = request.args.get("q", "").strip()

    return jsonify(
        {
            "filters": get_filters(),
            "tasks": build_task_list(genre=genre, platform=platform, query=query),
        }
    )


@app.route("/api/task/<task_id>")
def api_task(task_id: str):
    try:
        payload = get_task_payload(task_id)
        return jsonify(payload)
    except KeyError:
        return jsonify({"error": "Задание не найдено."}), 404


@app.route("/api/analytics")
def api_analytics():
    return jsonify(build_analytics_payload())


@app.route("/api/corpus_search")
def api_corpus_search():
    query = request.args.get("q", "").strip()
    genre = request.args.get("genre", "").strip()
    platform = request.args.get("platform", "").strip()

    results = search_corpus(query=query, genre=genre, platform=platform, limit=20)
    return jsonify({"results": results})


@app.route("/api/generate", methods=["POST"])
def api_generate():
    try:
        payload = request.get_json(silent=True) or {}
        task_id = safe_text(payload.get("task_id"))
        use_references = bool(payload.get("use_references", True))
        extra_instruction = safe_text(payload.get("extra_instruction", ""))

        if not task_id:
            return jsonify({"error": "Не передан идентификатор задания."}), 400

        result = generate_live_document(
            task_id=task_id,
            use_references=use_references,
            extra_instruction=extra_instruction,
        )
        return jsonify(result)

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Внутренняя ошибка генерации: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
