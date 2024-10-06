import aiohttp
import asyncio
import logging
from datetime import datetime

BASE_URL = "http://localhost:5000/api"

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def get_token(session, username, password):
    try:
        async with session.post(f"{BASE_URL}/login", json={"username": username, "password": password}) as response:
            if response.status == 200:
                data = await response.json()
                return data["access_token"]
            else:
                logger.error(f"Login failed. Status code: {response.status}")
                logger.error(f"Response: {await response.text()}")
                raise Exception("Login failed")
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        raise

async def mark_attendance(session, token, person_id, name, date, time):
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "person_id": person_id,
        "name": name,
        "date": date,
        "time": time
    }
    try:
        async with session.post(f"{BASE_URL}/mark_attendance", headers=headers, json=data) as response:
            if response.status == 201:
                logger.info(f"Successfully marked attendance for {name} (ID: {person_id})")
                return True
            else:
                logger.error(f"Failed to mark attendance. Status code: {response.status}")
                logger.error(await response.text())
                return False
    except aiohttp.ClientError as e:
        logger.error(f"Network error marking attendance: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error marking attendance: {str(e)}")
        return False

async def main():
    admin_username = "admin"
    admin_password = "admin"
    
    async with aiohttp.ClientSession() as session:
        try:
            token = await get_token(session, admin_username, admin_password)
            logger.info("Successfully obtained admin token")

            # Example usage of mark_attendance
            person_id = 1
            name = "John Doe"
            date = datetime.now().strftime("%Y-%m-%d")
            time = datetime.now().strftime("%H:%M:%S")

            success = await mark_attendance(session, token, person_id, name, date, time)
            if success:
                logger.info("Attendance marked successfully")
            else:
                logger.error("Failed to mark attendance")

        except aiohttp.ClientConnectorError:
            logger.error("Unable to connect to the server. Please check if the server is running and the BASE_URL is correct.")
        except asyncio.TimeoutError:
            logger.error("The request timed out. The server might be overloaded or unreachable.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")



if __name__ == "__main__":
    asyncio.run(main())