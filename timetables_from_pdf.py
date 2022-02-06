import pandas as pd 
import numpy as np

def timetable_to_datetime(timetable_file):
    #load original csv
    data = pd.read_csv(timetable_file, header=None)
    #remove empty columns
    data.dropna(axis='columns', inplace=True)
    #remove ellipses
    data = data[data[0]!='. . .'].reset_index(drop=True)
    #to datetime
    timetable = pd.DataFrame(columns=['timings'])
    for row in range(len(data.index)):
        #if next row is 2 or 4, the current contains a fraction
        current_cell = data.iloc[row,0]
        try:    next_cell = data.iloc[row+1,0]
        except:
            pass
        if len(current_cell) > 5:
            if next_cell == '2':
                to_append = current_cell[:5]+' 30'
                timetable._set_value(row,'timings',to_append)
            elif next_cell == '4':
                if current_cell[-1] == '1':
                    to_append = current_cell[:5]+' 15'
                elif current_cell[-1] == '3':
                    to_append = current_cell[:5]+' 45'
                timetable._set_value(row,'timings',to_append)
            else:
                print('error '+str(current_cell))
                print(next_cell)
        elif len(current_cell)==1:
            pass
        else:
            to_append = current_cell+' 00'
            timetable._set_value(row,'timings',to_append)
    timetable.reset_index(inplace=True, drop=True)

    #covert to datetime
    timetable.timings = pd.to_datetime(timetable.timings, format='%H %M %S')

    #find passing time
    if 'cir' in timetable_file:
        if 'eas' in timetable_file:
            timetable.timings += pd.to_timedelta('196.5S')
        else:
            timetable.timings += pd.to_timedelta('143.5S')
    elif 'bak' in timetable_file:
        if 'eas' in timetable_file:
            timetable.timings += pd.to_timedelta('19S')
        else:
            timetable.timings += pd.to_timedelta('81S')
    elif 'vic' in  timetable_file:
        if 'eas' in timetable_file:
            timetable.timings += pd.to_timedelta('16.5S')
        else:
            timetable.timings += pd.to_timedelta('101S')

    timetable.timings = timetable.timings.dt.time
    #save as csv
    timetable.to_csv(timetable_file[:-4]+'_.csv', index=False)

    ### ensure manually take overlap from post-midnight runs are put into next days csvs

timetable_to_datetime('timetables/cen_wes_sat.csv')