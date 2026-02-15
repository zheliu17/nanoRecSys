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

import json
import logging
import os

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, host="redis", port=6379, db=0):
        # Allow env var override
        host = os.getenv("REDIS_HOST", host)
        port = int(os.getenv("REDIS_PORT", port))
        try:
            self.host = host
            self.port = port
            self.db = db
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=True,
                socket_connect_timeout=1,
            )
            self.enabled = True
        except Exception:
            logger.warning("Redis init failed (will retry on connect), enabled=True")
            self.enabled = True

    async def get(self, key: str):
        if not self.enabled:
            return None
        try:
            val = await self.client.get(key)
            return json.loads(val) if val else None  # type: ignore
        except Exception:
            return None

    async def set(self, key: str, value: dict, expire=300):
        if not self.enabled:
            return
        try:
            val_str = json.dumps(value)
            await self.client.setex(key, expire, val_str)
        except Exception:
            pass
