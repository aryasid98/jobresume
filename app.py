-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Resumes table
CREATE TABLE IF NOT EXISTS resumes (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL UNIQUE,
    content_type TEXT,
    raw_text TEXT NOT NULL,
    extracted_data_json TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    file_path TEXT,
    storage_type TEXT DEFAULT 'local',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Jobs table
CREATE TABLE IF NOT EXISTS jobs (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    extracted_data_json TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    posted_date DATE DEFAULT CURRENT_DATE,
    source TEXT DEFAULT 'manual',
    posted_by TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Skills table
CREATE TABLE IF NOT EXISTS skills (
    id SERIAL PRIMARY KEY,
    skill_name TEXT NOT NULL UNIQUE,
    skill_name_lower TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    source TEXT NOT NULL,
    resume_count INTEGER DEFAULT 0,
    job_count INTEGER DEFAULT 0,
    first_seen TIMESTAMP DEFAULT NOW(),
    last_seen TIMESTAMP DEFAULT NOW()
);

-- Skill co-occurrence table
CREATE TABLE IF NOT EXISTS skill_cooccurrence (
    skill1_id INTEGER NOT NULL,
    skill2_id INTEGER NOT NULL,
    count INTEGER DEFAULT 1,
    PRIMARY KEY (skill1_id, skill2_id),
    FOREIGN KEY (skill1_id) REFERENCES skills(id),
    FOREIGN KEY (skill2_id) REFERENCES skills(id)
);

-- Resume embeddings table
CREATE TABLE IF NOT EXISTS resume_embeddings (
    id INTEGER PRIMARY KEY REFERENCES resumes(id) ON DELETE CASCADE,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Job embeddings table
CREATE TABLE IF NOT EXISTS job_embeddings (
    id INTEGER PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for similarity search
CREATE INDEX IF NOT EXISTS resume_embeddings_embedding_idx 
ON resume_embeddings USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS job_embeddings_embedding_idx 
ON job_embeddings USING ivfflat (embedding vector_cosine_ops);
