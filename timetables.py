import librosa
import pandas as pd
import datetime
import numpy as np
import os

#function to take start time and duration to return all trains in period
def trains_in_period(lines, start_time, duration='full'):
    """
    INPUT   - lines - array of lines in format ['cir_eas','bak_sou','vic_nor']
            - start_time - start of period, must be in full datetime
            - duration - duration of period, either 'full' (hour) or int in seconds

    OUTPUT  - train times in period, resetting time to zero in dictionary of train lines
    """
    #get end time
    if duration == 'full':
        end_time = start_time + datetime.timedelta(hours=1)
    elif isinstance(duration, int):
        end_time = start_time + datetime.timedelta(seconds=duration)
    else:
        print('Incorrect duration input')
        return

    #get day of week
    daysofweek = {0: 'mon', 1: 'tue', 2: 'wed', 3: 'thu', 4: 'fri', 5: 'sat', 6: 'sun'}
    dow = daysofweek[start_time.weekday()]
    dow_next = daysofweek[end_time.weekday()]
 
    #reset dates

    #load timetables
    timetables = {}
    timetables_day1 = {}
    directory = r'C:\Users\fredt\Documents\IIB Project work\Python\timetables'
    for filename in os.listdir(directory):
        if filename.endswith('_.csv'):
            for line in lines:
                if (line in filename) and (dow in filename):
                    timetables_day1[line] = pd.read_csv(directory+'\\'+filename)
                    #convert times to datetime at current date
                    timetables_day1[line].timings = pd.to_datetime(timetables_day1[line].timings + ' ' + str(start_time)[:10])
                    
    #check if period overruns day
    if dow != dow_next:
        timetables_day2 = {}
        for filename in os.listdir(directory):
            if filename.endswith('_.csv'):
                for line in lines:
                    if (line in filename) and (dow_next in filename):
                        timetables_day2[line] = pd.read_csv(directory+'\\'+filename)
                        #convert times to datetime at next day
                        timetables_day2[line].timings = pd.to_datetime(timetables_day1[line].timings + ' ' + str(start_time + pd.to_timedelta('1 day'))[:10])

        #cut to period if 2 days in new dict
        for key in timetables_day1.keys():
            timetables[key] = timetables_day1[key].timings[(timetables_day1[key].timings > start_time)] + timetables_day2[key].timings[(timetables_day2[key].timings < end_time)]

    else:
        #cut to period if single day to new dict
        for key in timetables_day1.keys():
            timetables[key] = timetables_day1[key].timings[(timetables_day1[key].timings > start_time) & (timetables_day1[key].timings < end_time)]
    

    #reset all times to zero at start of period - and make numpy arrays in seconds
    for key in timetables.keys():
        diff = timetables[key] - start_time
        timetables[key] = np.array(diff.dt.total_seconds())

    return timetables