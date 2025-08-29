from datetime import datetime
from zoneinfo import ZoneInfo

def get_sgt_time():
    return datetime.now(tz=ZoneInfo("Asia/Singapore"))