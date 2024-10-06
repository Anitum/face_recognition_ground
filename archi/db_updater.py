import asyncio
import aiohttp
import os
from datetime import datetime
from async_data_sender import get_token
import json

# Configuration
KNOWN_FACES_DIR = 'known_faces'
CHECK_INTERVAL = 10  # Check every 60 seconds
BASE_URL = 'http://localhost:5000'  # Replace with your actual base URL
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin"

session = None
token = None

async def initialize_session():
    global session, token
    if session is None or session.closed:
        session = aiohttp.ClientSession()
    if token is None:
        token = await get_token(session, ADMIN_USERNAME, ADMIN_PASSWORD)
    print(f"API token: {token}")

async def get_all_users():
    url = f"{BASE_URL}/api/all_users"
    headers = {"Authorization": f"Bearer {token}"}
    async with session.get(url, headers=headers) as response:
        if response.status == 200:
            return await response.json()
        else:
            print(f"Failed to get users. Status: {response.status}")
            return []

async def download_profile_picture(user_id):
    url = f"{BASE_URL}/api/known_face/{user_id}"
    headers = {"Authorization": f"Bearer {token}"}
    filename = f"{user_id}.jpg"
    filepath = os.path.join(KNOWN_FACES_DIR, filename)

    async with session.get(url, headers=headers) as response:
        if response.status == 200:
            with open(filepath, 'wb') as f:
                f.write(await response.read())
            print(f"Downloaded profile picture for user {user_id}")
            return True
        elif response.status == 404:
            print(f"No profile picture found for user {user_id}")
        else:
            print(f"Failed to download profile picture for user {user_id}. Status: {response.status}")
        return False

async def check_and_download_new_profiles_once(path):
    KNOWN_FACES_DIR = path
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
    
    print("Checking for new profiles...")

    known_users_file = 'known_users.json'
    await initialize_session()

    try:
        with open(known_users_file, 'r') as f:
            known_users = json.load(f)
    except FileNotFoundError:
        known_users = {}

    changes_detected = False

    try:
        all_users = await get_all_users()
        
        for user in all_users:
            user_id = str(user['id'])
            if user_id not in known_users or known_users[user_id] != user.get('profile_picture'):
                if await download_profile_picture(user_id):
                    known_users[user_id] = user.get('profile_picture')
                    changes_detected = True

        # Save updated known_users
        with open(known_users_file, 'w') as f:
            json.dump(known_users, f)

    except aiohttp.ClientError as e:
        print(f"Network error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    print("Done checking for new profiles. Changes detected: ", changes_detected)
    return changes_detected

async def main():

    while True:
        try:
            await initialize_session()
            await check_and_download_new_profiles_once()
        finally:
            if session and not session.closed:
                await session.close()
        await asyncio.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())