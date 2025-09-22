# pipelines/utils.py

import redis
import pandas as pd
import pickle

# Import from the same package
from .config import REDIS_HOST, REDIS_PORT

def get_redis():
    """
    Establishes and returns a Redis connection.
    Raises a ConnectionError if the connection fails.
    """
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        # The ping command is a great way to verify the connection
        r.ping()
        return r
    except redis.exceptions.ConnectionError as e:
        raise redis.exceptions.ConnectionError(
            f"Could not connect to Redis at {REDIS_HOST}:{REDIS_PORT}. "
            f"Please ensure the Redis service is running and accessible."
        ) from e

def df_to_bytes(df: pd.DataFrame) -> bytes:
    """
    Converts a Pandas DataFrame to a byte string using pickle.
    This is necessary for storing the dataframe in a key-value store like Redis.
    """
    return pickle.dumps(df)
