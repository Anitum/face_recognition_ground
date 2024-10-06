import asyncio
import aiohttp
import os
from datetime import datetime
import json

async def get_all_users(session, token, base_url):
    url = f"{base_url}/api/all_users"
    headers = {"Authorization": f"Bearer {token}"}
    async with session.get(url, headers=headers) as response:
        if response.status == 200:
            return await response.json()
        else:
            print(f"Failed to get users. Status: {response.status}")
            return []

async def download_profile_picture(session, token, base_url, known_faces_dir, user_id):
    url = f"{base_url}/api/known_face/{user_id}"
    headers = {"Authorization": f"Bearer {token}"}
    filename = f"{user_id}.jpg"
    filepath = os.path.join(known_faces_dir, filename)

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

async def check_and_download_new_profiles(session, token, base_url, known_faces_dir):
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
    
    print("Checking for new profiles...")

    known_users_file = os.path.join(known_faces_dir, 'known_users.json')

    try:
        with open(known_users_file, 'r') as f:
            known_users = json.load(f)
    except FileNotFoundError:
        known_users = {}

    changes_detected = False

    try:
        all_users = await get_all_users(session, token, base_url)
        
        for user in all_users:
            user_id = str(user['id'])
            if user_id not in known_users or known_users[user_id] != user.get('profile_picture'):
                if await download_profile_picture(session, token, base_url, known_faces_dir, user_id):
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

async def update_database(session, token, base_url, known_faces_dir):
    changes_detected = await check_and_download_new_profiles(session, token, base_url, known_faces_dir)
    return changes_detected