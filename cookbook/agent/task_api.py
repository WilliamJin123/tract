"""Task Management REST API implementation."""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, EmailStr, Field

# Models
class TaskStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    assignee: Optional[EmailStr] = None


class TaskUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    status: Optional[TaskStatus] = None
    assignee: Optional[EmailStr] = None


class Task(BaseModel):
    id: str
    title: str
    status: TaskStatus
    assignee: Optional[str]
    created_at: datetime
    updated_at: datetime


class PaginatedResponse(BaseModel):
    items: list[Task]
    total: int
    limit: int
    offset: int


# In-memory database
tasks_db: dict[str, dict] = {}

# Valid transitions
VALID_TRANSITIONS = {
    TaskStatus.TODO: {TaskStatus.IN_PROGRESS, TaskStatus.TODO},
    TaskStatus.IN_PROGRESS: {TaskStatus.DONE, TaskStatus.TODO},
    TaskStatus.DONE: {TaskStatus.TODO},
}

# API
app = FastAPI(title="Task Management API", version="1.0.0")


@app.post("/tasks", status_code=201, response_model=Task)
def create_task(task: TaskCreate) -> Task:
    """Create a new task."""
    task_id = str(uuid4())
    now = datetime.utcnow()

    task_data = {
        "id": task_id,
        "title": task.title,
        "status": TaskStatus.TODO,
        "assignee": task.assignee,
        "created_at": now,
        "updated_at": now,
    }

    tasks_db[task_id] = task_data
    return Task(**task_data)


@app.get("/tasks/{task_id}", response_model=Task)
def get_task(task_id: str) -> Task:
    """Get a specific task."""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    return Task(**tasks_db[task_id])


@app.get("/tasks", response_model=PaginatedResponse)
def list_tasks(
    status: Optional[TaskStatus] = None,
    assignee: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> PaginatedResponse:
    """List tasks with optional filtering and pagination."""
    filtered = list(tasks_db.values())

    if status:
        filtered = [t for t in filtered if t["status"] == status]

    if assignee:
        filtered = [t for t in filtered if t["assignee"] == assignee]

    total = len(filtered)
    items = [Task(**t) for t in filtered[offset : offset + limit]]

    return PaginatedResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@app.patch("/tasks/{task_id}", response_model=Task)
def update_task(task_id: str, update: TaskUpdate) -> Task:
    """Update a task (partial updates supported)."""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")

    task_data = tasks_db[task_id]

    # Validate status transition
    if update.status:
        current_status = TaskStatus(task_data["status"])
        if update.status not in VALID_TRANSITIONS.get(current_status, set()):
            raise HTTPException(
                status_code=409,
                detail=f"Cannot transition from {current_status} to {update.status}",
            )
        task_data["status"] = update.status

    # Update fields
    if update.title:
        task_data["title"] = update.title
    if update.assignee:
        task_data["assignee"] = update.assignee
    elif update.assignee is None and "assignee" in update.model_dump():
        task_data["assignee"] = None

    task_data["updated_at"] = datetime.utcnow()
    return Task(**task_data)


@app.delete("/tasks/{task_id}", status_code=204)
def delete_task(task_id: str) -> None:
    """Delete a task."""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    del tasks_db[task_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
