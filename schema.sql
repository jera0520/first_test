DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS analyses;

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL
);

CREATE TABLE analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    analysis_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    skin_type TEXT NOT NULL,
    recommendation_text TEXT NOT NULL,
    scores_json TEXT NOT NULL, 
    concerns_json TEXT NOT NULL,
    image_filename TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id)
);
