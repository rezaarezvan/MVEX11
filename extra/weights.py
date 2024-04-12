import os
import zipfile
import subprocess

BASE_DIR = "claudeslens/"
FILE_ID = "19D5XGSlvwqK7MsigmUUIlGMqFzQzv4FI"
FILE_NAME = "weights.zip"


def download_file(file_id, destination):
    try:
        subprocess.check_call(["gdown", "--id", file_id, "-O", destination])
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file {destination}: {e}")


def extract_zip_file(zip_path, target_path):
    with zipfile.ZipFile(zip_path, 'r') as file:
        file.extractall(path=target_path)


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    file_path = os.path.join(BASE_DIR, FILE_NAME)

    print(f"Downloading {FILE_NAME}...")
    download_file(FILE_ID, file_path)

    print(f"Extracting {FILE_NAME}...")
    extract_path = os.path.join(BASE_DIR)
    extract_zip_file(file_path, extract_path)

    os.remove(file_path)

    print("Setup completed.")


if __name__ == "__main__":
    main()
