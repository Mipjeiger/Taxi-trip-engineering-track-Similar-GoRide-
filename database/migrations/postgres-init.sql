CREATE TABLE IF NOT EXISTS trips (
    id SERIAL PRIMARY KEY,
    trip_id TEXT,
    pick_latitude DOUBLE PRECISION,
    pick_longitude DOUBLE PRECISION,
    dropoff_latitude DOUBLE PRECISION,
    dropoff_longitude DOUBLE PRECISION,
    fare_amount DOUBLE PRECISION,
    pickup_datetime TIMESTAMP
);