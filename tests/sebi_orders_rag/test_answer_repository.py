from __future__ import annotations

import unittest
from uuid import uuid4

from app.sebi_orders_rag.repositories.answers import AnswerRepository


class AnswerRepositoryTests(unittest.TestCase):
    def test_retries_retrieval_log_insert_on_deadlock(self) -> None:
        connection = _RetryingConnection(failures=1)
        repository = AnswerRepository(connection=connection)

        retrieval_id = repository.insert_retrieval_log(
            session_id=uuid4(),
            user_query="What did the Special Court hold?",
            route_mode="exact_lookup",
            query_intent="document_lookup",
            extracted_filters={},
        )

        self.assertEqual(retrieval_id, 1)
        self.assertEqual(connection.rollback_calls, 1)
        self.assertEqual(connection.execute_calls, 2)


class DeadlockDetected(Exception):
    pass


class _RetryingCursor:
    def __init__(self, connection: "_RetryingConnection") -> None:
        self._connection = connection

    def __enter__(self) -> "_RetryingCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, sql, params) -> None:
        del sql, params
        self._connection.execute_calls += 1
        if self._connection.failures_remaining > 0:
            self._connection.failures_remaining -= 1
            raise DeadlockDetected("simulated deadlock")

    def fetchone(self):
        return (1,)


class _RetryingConnection:
    def __init__(self, *, failures: int) -> None:
        self.failures_remaining = failures
        self.execute_calls = 0
        self.rollback_calls = 0

    def cursor(self) -> _RetryingCursor:
        return _RetryingCursor(self)

    def rollback(self) -> None:
        self.rollback_calls += 1


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
