ALTER TABLE chat_session_state
    ADD COLUMN IF NOT EXISTS clarification_context JSONB;
