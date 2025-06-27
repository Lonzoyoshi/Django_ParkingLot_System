# locustfile.py
from locust import HttpUser, task, between

class ParkingUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def get_stats(self):
        self.client.get("/parking/get_stats/")

    @task
    def get_records(self):
        self.client.get("/parking/get_records/")