import platform
sys_name = platform.system()
DEBUG = 1
if sys_name != "Windows":
    DEBUG = False
