import aiohttp
from datetime import datetime
import io

class ERPClient:
    BASE_URL = "http://3.232.128.127"
    LOGIN_ENDPOINT = f"{BASE_URL}/login"
    DASHBOARD_ENDPOINT = f"{BASE_URL}/dashboard"
    CREATE_PERSON_ENDPOINT = f"{BASE_URL}/api/create_person"
    RECORD_ATTENDANCE_ENDPOINT = f"{BASE_URL}/api/record_attendance"

    def __init__(self):
        self.session = None
        self.cookies = None

    async def login(self, username, password):
        self.session = aiohttp.ClientSession()
        async with self.session.post(self.LOGIN_ENDPOINT, data={"username": username, "password": password}, allow_redirects=False) as response:
            if response.status == 302:  # Successful login usually redirects
                self.cookies = response.cookies
                return "Login successful"
            else:
                raise Exception(f"Login failed with status code: {response.status}")

    async def get_dashboard(self):
        if not self.session:
            raise Exception("Not logged in")
        async with self.session.get(self.DASHBOARD_ENDPOINT, cookies=self.cookies) as response:
            if response.status == 200:
                return await response.text()
            else:
                raise Exception(f"Failed to get dashboard: {await response.text()}")

    async def create_person(self, name, arrival_time, photo_path):
        if not self.session:
            raise Exception("Not logged in")
        data = aiohttp.FormData()
        data.add_field('name', name)
        data.add_field('arrival_time', arrival_time)
        with open(photo_path, 'rb') as f:
            photo_data = f.read()
            data.add_field('photo', photo_data, filename='photo.jpg', content_type='image/jpeg')
        async with self.session.post(self.CREATE_PERSON_ENDPOINT, data=data, cookies=self.cookies) as response:
            if response.status == 201:
                return await response.json()
            else:
                raise Exception(f"Failed to create person: {await response.text()}")

    async def record_attendance(self, person_id, timestamp):
        if not self.session:
            raise Exception("Not logged in")
        data = {
            "person_id": person_id,
            "timestamp": timestamp.isoformat()
        }
        async with self.session.post(self.RECORD_ATTENDANCE_ENDPOINT, json=data, cookies=self.cookies) as response:
            if response.status in (200, 201):
                return await response.json()
            else:
                raise Exception(f"Failed to record attendance: {await response.text()}")

    async def close(self):
        if self.session:
            await self.session.close()