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

import orjson
import logging
import os

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, host="redis", port=6379, db=0):
        host = os.getenv("REDIS_HOST", host)
        port = int(os.getenv("REDIS_PORT", port))

        self.host = host
        self.port = port
        self.db = db
        self.enabled = True
        self.available = False
        self.error_count = 0

        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )

    async def ping(self) -> bool:
        if not self.enabled:
            self.available = False
            return False
        try:
            pong = await self.client.ping()  # type: ignore
            self.available = bool(pong)
            return self.available
        except Exception as e:
            self.available = False
            self.error_count += 1
            logger.warning("Redis ping failed at %s:%s: %s", self.host, self.port, e)
            return False

    async def get(self, key: str):
        if not self.enabled:
            return None
        try:
            val = await self.client.get(key)
            self.available = True
            return orjson.loads(val) if val else None  # type: ignore[arg-type]
        except Exception as e:
            self.available = False
            self.error_count += 1
            logger.debug("Redis GET failed for key=%s: %s", key, e)
            return None

    async def set(self, key: str, value: dict, expire=300):
        if not self.enabled:
            return
        try:
            val_str = orjson.dumps(value)
            await self.client.setex(key, expire, val_str)
            self.available = True
        except Exception as e:
            self.available = False
            self.error_count += 1
            logger.debug("Redis SET failed for key=%s: %s", key, e)

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "host": self.host,
            "port": self.port,
            "error_count": self.error_count,
        }
