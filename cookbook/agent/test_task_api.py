"""Tests for task management API."""

import pytest
from fastapi.testclient import TestClient
from task_api import app, tasks_db, TaskStatus


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_db():
    """Clear database before each test."""
    tasks_db.clear()
    yield
    tasks_db.clear()


class TestCreateTask:
    """Test task creation."""

    def test_create_task_minimal(self, client):
        """Create task with title only."""
        response = client.post("/tasks", json={"title": "Write docs"})
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Write docs"
        assert data["status"] == "todo"
        assert data["assignee"] is None
        assert "id" in data
        assert "created_at" in data

    def test_create_task_with_assignee(self, client):
        """Create task with assignee."""
        response = client.post(
            "/tasks",
            json={"title": "Code review", "assignee": "alice@example.com"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["assignee"] == "alice@example.com"

    def test_create_task_invalid_email(self, client):
        """Reject invalid email."""
        response = client.post(
            "/tasks", json={"title": "Task", "assignee": "not-an-email"}
        )
        assert response.status_code == 422

    def test_create_task_empty_title(self, client):
        """Reject empty title."""
        response = client.post("/tasks", json={"title": ""})
        assert response.status_code == 422

    def test_create_task_title_too_long(self, client):
        """Reject title over 200 chars."""
        response = client.post("/tasks", json={"title": "x" * 201})
        assert response.status_code == 422


class TestGetTask:
    """Test task retrieval."""

    def test_get_existing_task(self, client):
        """Retrieve existing task."""
        # Create
        create_resp = client.post("/tasks", json={"title": "Test task"})
        task_id = create_resp.json()["id"]

        # Get
        get_resp = client.get(f"/tasks/{task_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == task_id

    def test_get_nonexistent_task(self, client):
        """Return 404 for missing task."""
        response = client.get("/tasks/nonexistent-id")
        assert response.status_code == 404


class TestListTasks:
    """Test task listing."""

    def test_list_empty(self, client):
        """List returns empty array initially."""
        response = client.get("/tasks")
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_with_pagination(self, client):
        """Pagination works correctly."""
        # Create 5 tasks
        for i in range(5):
            client.post("/tasks", json={"title": f"Task {i}"})

        # Get with limit
        response = client.get("/tasks?limit=2&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2
        assert data["total"] == 5
        assert data["limit"] == 2
        assert data["offset"] == 0

    def test_list_filter_by_status(self, client):
        """Filter tasks by status."""
        task1 = client.post("/tasks", json={"title": "Task 1"}).json()
        task2 = client.post("/tasks", json={"title": "Task 2"}).json()

        # Mark task1 as done (must go todo → in_progress → done)
        client.patch(f"/tasks/{task1['id']}", json={"status": "in_progress"})
        client.patch(f"/tasks/{task1['id']}", json={"status": "done"})

        # List only done tasks
        response = client.get("/tasks?status=done")
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["id"] == task1["id"]

    def test_list_filter_by_assignee(self, client):
        """Filter tasks by assignee."""
        client.post(
            "/tasks",
            json={"title": "Task 1", "assignee": "alice@example.com"},
        )
        client.post("/tasks", json={"title": "Task 2"})

        response = client.get("/tasks?assignee=alice@example.com")
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1


class TestUpdateTask:
    """Test task updates."""

    def test_update_title(self, client):
        """Update task title."""
        task = client.post("/tasks", json={"title": "Old title"}).json()

        response = client.patch(
            f"/tasks/{task['id']}", json={"title": "New title"}
        )
        assert response.status_code == 200
        assert response.json()["title"] == "New title"

    def test_update_status_todo_to_in_progress(self, client):
        """Transition from todo to in_progress."""
        task = client.post("/tasks", json={"title": "Task"}).json()

        response = client.patch(
            f"/tasks/{task['id']}", json={"status": "in_progress"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "in_progress"

    def test_update_status_in_progress_to_done(self, client):
        """Transition from in_progress to done."""
        task = client.post("/tasks", json={"title": "Task"}).json()
        client.patch(f"/tasks/{task['id']}", json={"status": "in_progress"})

        response = client.patch(
            f"/tasks/{task['id']}", json={"status": "done"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "done"

    def test_update_status_done_to_todo(self, client):
        """Can transition done back to todo."""
        task = client.post("/tasks", json={"title": "Task"}).json()
        client.patch(f"/tasks/{task['id']}", json={"status": "in_progress"})
        client.patch(f"/tasks/{task['id']}", json={"status": "done"})

        response = client.patch(
            f"/tasks/{task['id']}", json={"status": "todo"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "todo"

    def test_update_invalid_transition(self, client):
        """Reject invalid status transitions."""
        task = client.post("/tasks", json={"title": "Task"}).json()

        # Try to go directly from todo to done (should fail)
        response = client.patch(
            f"/tasks/{task['id']}", json={"status": "done"}
        )
        assert response.status_code == 409

    def test_update_nonexistent_task(self, client):
        """Return 404 when updating missing task."""
        response = client.patch("/tasks/nonexistent", json={"title": "New"})
        assert response.status_code == 404

    def test_update_assignee(self, client):
        """Update assignee."""
        task = client.post("/tasks", json={"title": "Task"}).json()

        response = client.patch(
            f"/tasks/{task['id']}", json={"assignee": "bob@example.com"}
        )
        assert response.status_code == 200
        assert response.json()["assignee"] == "bob@example.com"


class TestDeleteTask:
    """Test task deletion."""

    def test_delete_task(self, client):
        """Delete a task."""
        task = client.post("/tasks", json={"title": "Task"}).json()

        response = client.delete(f"/tasks/{task['id']}")
        assert response.status_code == 204

        # Verify deleted
        get_resp = client.get(f"/tasks/{task['id']}")
        assert get_resp.status_code == 404

    def test_delete_nonexistent_task(self, client):
        """Return 404 when deleting missing task."""
        response = client.delete("/tasks/nonexistent")
        assert response.status_code == 404
