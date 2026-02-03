from locust import HttpUser, task, between, events


class RecSysUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def get_recommendations(self):
        # 1. Random user ID around the range of known users (e.g. 1-138000)
        # Using a smaller range ensures some cache hits if Redis is on
        import random

        user_id = random.randint(1, 1000)

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
