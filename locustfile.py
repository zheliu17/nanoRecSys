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

from locust import HttpUser, task, between, events


class RecSysUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def get_recommendations(self):
        import random

        # Simulate 70% "Hot" users (Cache Hits) and 30% "Cold" users (Cache Misses)
        if random.random() < 0.7:
            # Hot users: Reuse a small set of IDs (e.g., 1-1000)
            user_id = random.randint(1, 1000)
        else:
            # Cold users: Pick from the rest of your 128k dataset (e.g., 1001-128000)
            # This forces the system to run Embedding + Retrieval + Ranking
            user_id = random.randint(1001, 128000)

        payload = {
            "user_id": user_id,
            "k": 10,
            "explain": False,
            "include_history": False,
        }

        with self.client.post(
            "/recommend", json=payload, catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
                try:
                    data = response.json()
                    if "debug_timing" in data and data["debug_timing"]:
                        timings = data["debug_timing"]
                        # Report to Locust as "custom" request
                        if "embedding" in timings:
                            events.request.fire(
                                request_type="DB",
                                name="Embedding",
                                response_time=timings["embedding"],
                                response_length=0,
                                exception=None,
                            )
                        if "retrieval" in timings:
                            events.request.fire(
                                request_type="DB",
                                name="Retrieval",
                                response_time=timings["retrieval"],
                                response_length=0,
                                exception=None,
                            )
                        if "ranking" in timings:
                            events.request.fire(
                                request_type="DB",
                                name="Ranking",
                                response_time=timings["ranking"],
                                response_length=0,
                                exception=None,
                            )
                        if "total" in timings:
                            events.request.fire(
                                request_type="DB",
                                name="Server_Processing",
                                response_time=timings["total"],
                                response_length=0,
                                exception=None,
                            )
                except Exception as e:
                    print(f"Error parsing timing: {e}")
            else:
                response.failure(f"Status code: {response.status_code}")
