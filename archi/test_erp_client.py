import asyncio
from erp_client import ERPClient
from datetime import datetime, timedelta
import os

async def test_login_success(erp_client):
    try:
        result = await erp_client.login('admin', 'admin')
        assert result == "Login successful"
        print("Login success test passed")
    except Exception as e:
        print(f"Login success test failed: {str(e)}")

async def test_get_dashboard(erp_client):
    try:
        dashboard = await erp_client.get_dashboard()
        assert "Dashboard" in dashboard
        print("Get dashboard test passed")
    except Exception as e:
        print(f"Get dashboard test failed: {str(e)}")

async def test_create_person(erp_client):
    try:
        # Ensure you have a test image file
        test_image_path = '/home/mridul/face_regonition_ground/known_faces/deepika_dhamaja.png'
        if not os.path.exists(test_image_path):
            raise FileNotFoundError(f"Test image file not found: {test_image_path}")
        
        response = await erp_client.create_person('Test Person', '09:00', test_image_path)
        assert 'id' in response
        assert 'name' in response
        print(f"Create person test passed. Response: {response}")
        return response['id']
    except Exception as e:
        print(f"Create person test failed: {str(e)}")
        return None

async def test_record_attendance(erp_client, person_id):
    try:
        # Use yesterday's date to avoid the "already recorded" message
        timestamp = datetime.now() - timedelta(days=1)
        response = await erp_client.record_attendance(person_id, timestamp)
        assert 'message' in response
        print(f"Record attendance test passed. Response: {response}")
    except Exception as e:
        print(f"Record attendance test failed: {str(e)}")

async def run_tests():
    erp_client = ERPClient()
    try:
        await test_login_success(erp_client)
        await test_get_dashboard(erp_client)
        person_id = await test_create_person(erp_client)
        if person_id:
            await test_record_attendance(erp_client, person_id)
        else:
            print("Skipping attendance test due to person creation failure")
    finally:
        await erp_client.close()

if __name__ == "__main__":
    asyncio.run(run_tests())