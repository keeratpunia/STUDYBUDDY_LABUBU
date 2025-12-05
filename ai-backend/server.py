# server.py
import os
from typing import List, Optional
import uuid
import threading
from fastapi import FastAPI
from pydantic import BaseModel

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Emotion_Behavior.attentiveORdistracted_copy import run_attentiveness_check
                
jobs = {}

from ai_core import (
    generate_single_question,
    check_answer,
    solve_doubt,
    summarize_notes,
    ingest_pdf,
)

app = FastAPI(title="StudyBuddy AI Backend")


class QuizRequest(BaseModel):
    topic: str
    difficulty: str = "Medium"
    num_questions: int = 5


class CheckAnswerRequest(BaseModel):
    question_data: dict
    user_answer: str


class DoubtRequest(BaseModel):
    question: str
    last_answer: Optional[str] = ""


class SummaryRequest(BaseModel):
    mode: str = "Detailed"


class IngestRequest(BaseModel):
    path: str


def run_summary(job_id: str, mode: str):
    try:
        result = summarize_notes(mode)
        jobs[job_id] = {"status": "done", "result": result}
    except Exception as e:
        jobs[job_id] = {"status": "error", "result": str(e)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    pages = ingest_pdf(req.path)
    return {"ok": True, "pages": pages, "path": req.path}


@app.post("/quiz")
def quiz(req: QuizRequest):
    if req.difficulty not in ["Easy", "Medium", "Hard"]:
        return {"ok": False, "error": "Invalid difficulty"}

    if len(req.topic.strip()) < 2:
        return {"ok": False, "error": "Topic too short"}

    used_questions = []
    questions = []

    for _ in range(req.num_questions):
        q = generate_single_question(req.topic, req.difficulty, used_questions)
        if q:
            used_questions.append(q["question"].strip())
            questions.append(q)

    if not questions:
        return {"ok": False, "error": "No questions could be generated"}

    return {"ok": True, "questions": questions}


@app.post("/quiz/check")
def quiz_check(req: CheckAnswerRequest):
    correct = check_answer(req.question_data, req.user_answer)
    return {"ok": True, "correct": correct}


@app.post("/doubt")
def doubt(req: DoubtRequest):
    answer = solve_doubt(req.question, last_answer=req.last_answer or "")
    return {"ok": True, "answer": answer}


@app.post("/summarize/start")
def summarize_start(req: SummaryRequest):
    mode = "Brief" if req.mode.lower().startswith("brief") else "Detailed"
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "result": ""}

    t = threading.Thread(target=run_summary, args=(job_id, mode))
    t.start()

    return {"ok": True, "job_id": job_id}


@app.get("/summarize/status/{job_id}")
def summarize_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"ok": False, "error": "Invalid job_id"}

    if job["status"] == "done":
        return {"ok": True, "status": "done", "summary": job["result"]}

    if job["status"] == "error":
        return {"ok": False, "status": "error", "error": job["result"]}

    return {"ok": True, "status": "processing"}
@app.post("/attentive")
def run_attentive():
    return { "ok": True, **run_attentiveness_check() }

if __name__ == "__main__":
    port = int(os.environ.get("AI_PORT", "8000"))
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port)
