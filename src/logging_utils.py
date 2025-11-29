import time

def print_log(message: str) -> None:
    current_time = time.localtime()
    print(time.strftime("%Y-%m-%d %H:%M:%S", current_time) + " " + str(message))
