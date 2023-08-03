import socket
from datetime import datetime

hostname = None
ip_address = None

def get_log_head():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    global hostname
    global ip_address
    if hostname == None or ip_address == None:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
    result = f"[{current_time}] [{hostname}] [{ip_address}]"
    return result
