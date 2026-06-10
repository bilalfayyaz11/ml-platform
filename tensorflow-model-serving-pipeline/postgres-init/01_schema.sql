CREATE TABLE IF NOT EXISTS predictions (
    id          SERIAL PRIMARY KEY,
    features    JSONB        NOT NULL,
    prediction  INTEGER      NOT NULL,
    confidence  NUMERIC(6,4) NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
