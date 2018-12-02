r2 = movavg.rolling(window=500, win_type='boxcar')
event_part = part[0:500]
event_part = event_part[lambda df: df.v1 > 0.5]