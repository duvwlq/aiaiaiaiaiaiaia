import subprocess
import sys


def install_latest_package(package_name):
    """
    PyPI에서 해당 패키지의 최신 버전을 검색하고 설치하는 함수.
    :param package_name: 패키지 이름 (예: 'ultralytics', 'torch', 'torchvision')
    """
    try:
        # 최신 버전 검색 (pip search는 더 이상 지원되지 않으므로 설치로 직접 처리)
        print(f"Searching and installing the latest version of '{package_name}'...")

        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", package_name], check=True)

        print(f"The latest version of '{package_name}' has been successfully installed!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during the installation of '{package_name}': {e}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")


if __name__ == "__main__":
    # 설치하고자 하는 패키지 이름을 여기에 입력
    packages = ["ultralytics", "torch", "torchvision"]

    for pkg in packages:
        install_latest_package(pkg)
