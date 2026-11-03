CREATE TABLE IF NOT EXISTS image_analysis (
    image_id BIGSERIAL PRIMARY KEY,
    image_name TEXT NOT NULL,
    analyzed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_version TEXT,
    total_objects INTEGER NOT NULL DEFAULT 0,
    processing_time_ms NUMERIC
);

CREATE TABLE IF NOT EXISTS object_detection (
    detection_id BIGSERIAL PRIMARY KEY,
    image_id BIGINT NOT NULL REFERENCES image_analysis(image_id) ON DELETE CASCADE,
    object_name TEXT NOT NULL,
    class_id INTEGER,
    confidence NUMERIC,
    x1 NUMERIC,
    y1 NUMERIC,
    x2 NUMERIC,
    y2 NUMERIC,
    box_width NUMERIC,
    box_height NUMERIC
);

CREATE INDEX IF NOT EXISTS idx_detection_object_name
    ON object_detection(object_name);
CREATE INDEX IF NOT EXISTS idx_image_analysis_analyzed_at
    ON image_analysis(analyzed_at);

-- Example business query
SELECT object_name, COUNT(*) AS detected_count,
       ROUND(AVG(confidence)::numeric, 3) AS average_confidence
FROM object_detection
GROUP BY object_name
ORDER BY detected_count DESC;
