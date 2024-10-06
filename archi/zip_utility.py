import os
import zipfile
import time
from datetime import datetime
import schedule
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def get_google_drive_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('/home/mridul/ERP_attendence_system/app/model/archi/cred/cred.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def zip_folder(source_folder, output_filename):
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_folder)
                zipf.write(file_path, arcname)

def upload_to_drive(service, file_path, file_name):
    file_metadata = {'name': file_name}
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f'File ID: {file.get("id")}')
    return file.get('id')

def zip_and_upload():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "/home/mridul/ERP_attendence_system/app/model/output_images"
    zip_filename = f"/home/mridul/ERP_attendence_system/zipped_output_{timestamp}.zip"
    
    zip_folder(output_dir, zip_filename)
    print(f"Zipped output folders to {zip_filename}")

    service = get_google_drive_service()
    file_id = upload_to_drive(service, zip_filename, os.path.basename(zip_filename))
    print(f"Uploaded zipped file to Google Drive with ID: {file_id}")

    # Optionally, delete the local zip file after uploading
    os.remove(zip_filename)
    print(f"Deleted local zip file: {zip_filename}")

    # Optionally, clear the output folders after zipping
    import shutil
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, "faces"))
    os.makedirs(os.path.join(output_dir, "frames"))

def main():
    print("Starting folder zipping and Google Drive upload utility...")
    
    # Schedule the task to run at 1:00 PM and 6:00 PM
    schedule.every().day.at("13:00").do(zip_and_upload)
    schedule.every().day.at("18:00").do(zip_and_upload)

    # Run once immediately on start
    zip_and_upload()
    

    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()