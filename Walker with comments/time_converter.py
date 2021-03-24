def timespan_format(timespan):
    timespan = round(timespan)
    h_full = timespan / (60*60)
    h = timespan // (60*60)
    m = (timespan % (60*60)) // 60
    s = timespan % 60
    time = f"{h:02}:{m:02}:{s:02}"
    print(time)
    if h_full > 24:
        d = timespan // (60*60*24)
        h = (timespan % (60*60*24)) // (60*60)
        time = f"{d:02}:{h:02}:{m:02}:{s:02}"
    print(time)
    return time

timespan_format(303601)
test = False
if test:
    assert timespan_format(60) == "00:01:00"
    assert timespan_format(60*60) == "01:00:00"
    assert timespan_format(60*60+1) == "01:00:01"
    assert timespan_format(60*60+60) == "01:01:00"
    assert timespan_format(24*60*60+1) == "01:00:00:01"
    assert timespan_format(303601) == "03:12:20:01" # 3 days 12 hours and 20 minutes 1 second
