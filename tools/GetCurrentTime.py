import datetime

def get_current_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
