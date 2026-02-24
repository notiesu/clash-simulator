#remove all logs

import os
log_dir = "replays/logs"

if __name__ == "__main__":
    if os.path.exists(log_dir):
        for filename in os.listdir(log_dir):
            file_path = os.path.join(log_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed log file: {file_path}")
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")