import subprocess


def check_nvidia_driver():
    required_version = "570"
    try:
        version = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        if not version.startswith(required_version):
            yellow = "\033[93m"
            red = "\033[91m"
            reset = "\033[0m"
            print(
                f"{yellow}Warning: NVIDIA driver version is {red}{version}{yellow}, expected {required_version}.x{reset}"
            )
            print(
                "For more info, see: https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html"
            )
    except Exception as e:
        print(f"Could not determine NVIDIA driver version: {e}")