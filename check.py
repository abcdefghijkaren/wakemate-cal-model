from datetime import datetime

dt = datetime.now()  # 或是某個你要檢查的 datetime 物件

# 檢查是否為 timezone-aware
if dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None:
    print("timezone-aware")
else:
    print("naive datetime（沒有時區）")