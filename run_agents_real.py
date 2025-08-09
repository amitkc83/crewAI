"""
run_agents_real.py
Real GPT-5 Agent Runner for FaceAttend (replaces simulated progress with actual model execution)

Features:
 - Reads faceattend_ai_agent_command_board.json
 - Spawns one thread / worker per agent
 - For each non-N/A task, asks the model for a multi-step plan and (optionally) code
 - Sends progress updates to dashboard /api/update

Requirements:
    pip install openai requests
Set env var: OPENAI_API_KEY
"""

import os
import json
import time
import math
import random
import threading
import logging
from typing import Dict, Any, List
import requests
import openai

# ---------- CONFIG ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in environment")

# Model to use (change if you have a specific GPT-5 model name)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

# Path to command board (created earlier)
COMMAND_BOARD_PATH = "faceattend_ai_agent_command_board.json"

# Dashboard endpoint
DASHBOARD_UPDATE_URL = os.getenv("DASHBOARD_UPDATE_URL", "http://localhost:4000/api/update")

# Timeout & retry settings
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

# Concurrency (max agent threads)
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "6"))

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agent-runner")

# OpenAI client
openai.api_key = OPENAI_API_KEY

# ---------- HELPER: send progress to dashboard ----------
def send_progress(agent_name: str, phase: str, task: str, percent: int, message: str) -> bool:
    payload = {
        "agent": agent_name,
        "phase": phase,
        "task": task,
        "percent": max(0, min(100, int(percent))),
        "message": message
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(DASHBOARD_UPDATE_URL, json=payload, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            logger.info(f"Sent update: {payload}")
            return True
        except Exception as e:
            logger.warning(f"Failed to send update (attempt {attempt}): {e}")
            time.sleep(RETRY_BACKOFF ** attempt)
    logger.error("Giving up sending update after retries: %s", payload)
    return False

# ---------- HELPER: ask model for multi-step plan ----------
def request_plan_from_model(agent_name: str, role_prompt: str, phase: str, task: str, max_steps:int=4) -> List[Dict[str, Any]]:
    """
    Ask the model to produce a concise step-list for the given task.
    The model is instructed to return JSON like:
    { "steps": [ {"title":"...", "details":"...","deliverable":"..."} , ... ] }
    """
    system_msg = f"You are {agent_name}. {role_prompt}\n\nBehavior: Keep replies brief and return only JSON with a 'steps' list. Each step object must include 'title', 'description', and 'expected_percent' (integer). Return exactly valid JSON, nothing else."

    user_msg = (
        f"Task: {task}\n"
        f"Phase: {phase}\n"
        f"Provide a plan broken into {max_steps} logical steps for implementation and testing. "
        "For each step return: title, description, expected_percent (cumulative percentage completion for this step). "
        "Make expected_percent increase progressively and end with 100 for the final step."
    )

    for attempt in range(1, MAX_RETRIES+1):
        try:
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role":"system","content":system_msg},
                    {"role":"user","content":user_msg}
                ],
                temperature=0.2,
                max_tokens=800,
                n=1
            )
            text = resp["choices"][0]["message"]["content"].strip()
            # parse JSON
            parsed = json.loads(text)
            if "steps" in parsed and isinstance(parsed["steps"], list):
                # Basic validation & normalize
                steps = []
                last_pct = 0
                for s in parsed["steps"]:
                    title = s.get("title", "")[:240]
                    desc = s.get("description", "")[:2000]
                    pct = int(s.get("expected_percent", 0))
                    if pct <= last_pct:
                        pct = min(100, last_pct + max(1, (100-last_pct)//(max_steps)))
                    last_pct = pct
                    steps.append({"title": title, "description": desc, "expected_percent": pct})
                # Ensure last step 100
                if steps:
                    steps[-1]["expected_percent"] = 100
                return steps
            else:
                raise ValueError("Model output missing 'steps'")
        except Exception as e:
            logger.warning(f"Plan request failed (attempt {attempt}): {e}")
            time.sleep(RETRY_BACKOFF ** attempt)
    raise RuntimeError("Failed to obtain plan from model after retries")

# ---------- HELPER: optionally generate a deliverable for a step ----------
def request_deliverable(agent_name: str, role_prompt: str, step_title: str, step_description: str, task: str) -> Dict[str, Any]:
    """
    Ask model to produce a short deliverable for a single step. Limit output size.
    Returns dict with keys: { 'summary': str, 'artifact': str (optional short code or sample) }
    """
    system_msg = f"You are {agent_name}. {role_prompt}\n\nBehavior: Provide a concise summary and - if appropriate - a short code snippet or JSON artifact. Return JSON."
    user_msg = (
        f"Parent Task: {task}\n"
        f"Step: {step_title}\n"
        f"Description: {step_description}\n\n"
        "Provide a short summary (one or two sentences) and a short artifact field (code snippet or sample response) if relevant."
    )

    for attempt in range(1, MAX_RETRIES+1):
        try:
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role":"system","content":system_msg},
                    {"role":"user","content":user_msg}
                ],
                temperature=0.25,
                max_tokens=600,
                n=1
            )
            text = resp["choices"][0]["message"]["content"].strip()
            parsed = json.loads(text)
            return parsed
        except Exception as e:
            logger.warning(f"Deliverable request failed (attempt {attempt}): {e}")
            time.sleep(RETRY_BACKOFF ** attempt)
    return {"summary":"(failed to fetch deliverable)","artifact":""}

# ---------- Worker that runs per agent ----------
def agent_worker(agent_info: Dict[str, Any], role_prompt_map: Dict[str,str]):
    agent_name = agent_info["name"]
    role_prompt = role_prompt_map.get(agent_name, "")
    logger.info("Agent starting: %s", agent_name)

    # iterate phases/tasks
    for phase, task in agent_info["tasks"].items():
        if not task or task.strip().upper() == "N/A":
            continue

        try:
            # ask model for a plan (4 steps or fewer)
            steps = request_plan_from_model(agent_name, role_prompt, phase, task, max_steps=4)
            logger.info("%s plan received: %s", agent_name, [s["title"] for s in steps])
        except Exception as e:
            logger.error("%s failed to get plan for task '%s': %s", agent_name, task, e)
            # send a failed update and skip
            send_progress(agent_name, phase, task, 0, f"ERROR generating plan: {str(e)}")
            continue

        prev_pct = 0
        for step in steps:
            pct = int(step.get("expected_percent", prev_pct))
            title = step.get("title", "step")
            desc = step.get("description", "")

            # Announce starting step
            send_progress(agent_name, phase, task, prev_pct, f"Starting step: {title}")

            # Optionally ask the model to create a short deliverable for the step
            # Keep deliverable generation limited to avoid big token usage; you can toggle this behavior.
            try:
                deliverable = request_deliverable(agent_name, role_prompt, title, desc, task)
                summary = deliverable.get("summary", "") or title
                # send 'in-progress' with brief info
                send_progress(agent_name, phase, task, max(prev_pct, pct//2), f"{title} — {summary}")
            except Exception as e:
                logger.warning("Deliverable generation failed for %s: %s", agent_name, e)

            # Simulate local processing time (small sleep), realistic runs might run tests/builds etc.
            sleep_time = random.uniform(1.5, 4.0)
            time.sleep(sleep_time)

            # Mark step completion
            send_progress(agent_name, phase, task, pct, f"Completed step: {title}")

            prev_pct = pct

        # Ensure task marked 100% done
        send_progress(agent_name, phase, task, 100, "Task complete")

    logger.info("Agent finished: %s", agent_name)

# ---------- MAIN ----------
def main():
    with open(COMMAND_BOARD_PATH, "r") as f:
        board = json.load(f)

    # simple role prompts map (copy your earlier prompts here or load from file)
    role_prompts = {
        "Atlas": "You are Atlas, the Project Orchestrator. Coordinate tasks and produce actionable subtasks.",
        "Pixel": "You are Pixel, Flutter UI/UX specialist. Produce UI wireframes and Flutter widget snippets.",
        "Link": "You are Link, Flutter Integration Agent. Provide API service classes and integration steps.",
        "Nova": "You are Nova, Backend Architect. Produce Django models and REST API designs.",
        "Vision": "You are Vision, ML/Face Recognition specialist. Provide integration steps for liveness detection.",
        "Aegis": "You are Aegis, Security & Compliance Agent. Provide security best-practices and checklist items.",
        "Quill": "You are Quill, QA & Testing Agent. Output test-cases and testing procedures.",
        "Forge": "You are Forge, DevOps. Produce CI/CD and deployment steps."
    }

    # Spawn threads (bounded by MAX_WORKERS)
    threads = []
    sem = threading.BoundedSemaphore(value=MAX_WORKERS)

    def thread_target(a_info):
        with sem:
            agent_worker(a_info, role_prompts)

    for agent_info in board["agents"]:
        t = threading.Thread(target=thread_target, args=(agent_info,), daemon=False)
        threads.append(t)
        t.start()
        time.sleep(0.5)  # small stagger to avoid bursts

    # Wait for all threads to finish
    for t in threads:
        t.join()

    logger.info("All agents completed")

if __name__ == "__main__":
    main()
