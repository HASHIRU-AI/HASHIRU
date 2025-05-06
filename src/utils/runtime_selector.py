import platform
import GPUtil
from dotenv import load_dotenv
load_dotenv()

def detect_runtime_environment():
    os_type = platform.system().lower()
    gpus = GPUtil.getGPUs()
    if gpus:
        return "gpu"
    elif os_type in ["darwin", "linux"]:
        return "cpu-local"
    else:
        return "cloud-only"
