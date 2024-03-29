import os
import tarfile
import subprocess

BASE_DIR = "extra/datasets/SODA"
FILES = {
    "labeled_trainval.tar": "1oSJ0rbqNHLmlOOzpmQqXLDraCCQss4Q4",
    "labeled_test.tar": "1lA_XgKUHV9oc2LyP-WLvjhFtDHJ3lo6_"
}


def download_file(file_id, destination):
    try:
        subprocess.check_call(["gdown", file_id, "-O", destination])
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file {destination}: {e}")


def extract_tar_file(tar_path, target_path):
    with tarfile.open(tar_path) as file:
        file.extractall(path=target_path)


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    for filename, file_id in FILES.items():
        file_path = os.path.join(BASE_DIR, filename)

        print(f"Downloading {filename}...")
        download_file(file_id, file_path)

        print(f"Extracting {filename}...")
        extract_path = os.path.join(BASE_DIR, filename.replace(".tar", ""))
        extract_tar_file(file_path, extract_path)

        os.remove(file_path)

    print("Setup completed.")


if __name__ == "__main__":
    main()
