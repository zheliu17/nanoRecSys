# Copyright (c) 2026 Zhe Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import redis
import json
import os

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, host="redis", port=6379, db=0):
        # Allow env var override
        host = os.getenv("REDIS_HOST", host)
        port = int(os.getenv("REDIS_PORT", port))
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_connect_timeout=1,
            )
            self.client.ping()
            self.enabled = True
        except Exception:
            logger.warning("Redis not available, caching disabled.")
            self.enabled = False

    def get(self, key: str):
        if not self.enabled:
            return None
        try:
            val = self.client.get(key)
            return json.loads(val) if val else None  # type: ignore
        except Exception:
            return None

    def set(self, key: str, value: dict, expire=300):
        if not self.enabled:
            return
        try:
            self.client.setex(key, expire, json.dumps(value))
        except Exception:
            pass
