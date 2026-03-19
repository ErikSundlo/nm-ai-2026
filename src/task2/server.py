"""
Task 2 — Tripletex Accounting Agent
POST /solve  →  runs Claude agent against Tripletex sandbox, returns {"status": "completed"}

Expected request body (JSON):
{
    "prompt": "<task description in any supported language>",
    "tripletex_base_url": "https://...",
    "session_token": "<tripletex session token>",
    "attachments": [          // optional
        {
            "type": "base64",
            "media_type": "application/pdf",  // or image/jpeg etc.
            "data": "<base64-encoded content>"
        }
    ]
}
"""
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Request

from src.task2.agent import run_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="BradskiBeat — Tripletex Agent")
executor = ThreadPoolExecutor(max_workers=10)


@app.middleware("http")
async def log_all_requests(request: Request, call_next):
    log.info("INCOMING %s %s from %s", request.method, request.url.path, request.client)
    response = await call_next(request)
    log.info("RESPONSE %s %s -> %s", request.method, request.url.path, response.status_code)
    return response


@app.get("/solve ")
@app.get("/solve")
def solve_ping():
    return {"status": "ok"}


@app.post("/solve ")
@app.post("/solve")
async def solve(request: Request):
    body = await request.json()
    log.info("Raw payload: %s", json.dumps(body, ensure_ascii=False)[:500])

    # Extract fields
    prompt = body.get("prompt") or ""
    creds = body.get("tripletex_credentials") or {}
    base_url = creds.get("base_url") or body.get("tripletex_base_url") or ""
    token = creds.get("session_token") or body.get("session_token") or ""

    attachments_raw = body.get("files") or body.get("attachments") or []
    attachments = [
        {"type": a.get("type", "base64"), "media_type": a["media_type"], "data": a["data"]}
        for a in attachments_raw
        if "data" in a
    ] or None

    if not prompt:
        log.warning("Could not find prompt in payload keys: %s", list(body.keys()))
    if not base_url:
        log.warning("Could not find base_url in payload keys: %s", list(body.keys()))

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor, lambda: run_agent(
                prompt=prompt,
                tripletex_base_url=base_url,
                session_token=token,
                attachments=attachments,
            )
        )
        log.info("Task completed successfully")
        return {"status": "completed"}
    except Exception as e:
        log.exception("Agent failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
@app.get("/")
def health():
    return {"status": "ok", "team": "BradskiBeat"}


@app.post("/")
async def solve_root(request: Request):
    log.info("POST / received — redirecting to solve handler")
    return await solve(request)
