import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

#functions to take EB and WB trains at stations, to calculate expected passing time
#input times past hour in minutes
def find_central_line_passing_times(eastbound, westbound, start_time):
    #if train is eastbound - we add 196.5 seconds to time departing MAR
    #if train is westbound - we add 143.5 seconds to time departing HOL  
    east = [round(((i-start_time)*60+196.5)*2)/2 for i in eastbound]
    west = [round(((i-start_time)*60+143.5)*2)/2 for i in westbound]
    return east, west
def find_bakerloo_line_passing_times(southbound, northbound, start_time):
    #if train is southbound - we add 19 seconds to time departing OXC
    #if train is northbound - we add 81 seconds to time departing PIC  
    south = [round(((i-start_time)*60+19)*2)/2 for i in southbound]
    north = [round(((i-start_time)*60+81)*2)/2 for i in northbound]
    
    return south, north
def find_victoria_line_passing_times(southbound, northbound, start_time):
    #if train is southbound - we add 16.5 seconds to time departing OXC
    #if train is northbound - we add 101 seconds to time departing GNP
    south = [round(((i-start_time)*60+16.5)*2)/2 for i in southbound]
    north = [round(((i-start_time)*60+101)*2)/2 for i in northbound]
    
    return south, north

def plot_hour10_trains(k, lines, plot):
    #expected times - central
    expected_times_cen_eastbound = [0.75,4,7.5,10.75,14,17.5,20.75,24,27.5,30.75,34,37.5,40.75,44,47.5,50.75,54,57.5,60.75,64,67.5,70.75]
    expected_times_cen_westbound = [2.25,5.75,9,12.25,15.75,19,22.25,25.75,29,32.25,35.75,39,42.25,45.75,49,52.25,55.75,59,62.25,65.75,69]
    z_cen_east, z_cen_west = find_central_line_passing_times(eastbound=expected_times_cen_eastbound, westbound=expected_times_cen_westbound, start_time=32851/3600)   
    x_cen = np.linspace(0,3600, 7201)
    y_cen_east = [0] * len(x_cen)
    y_cen_west = [0] * len(x_cen)
    for i, j in enumerate(x_cen):
        if j in z_cen_east:
            y_cen_east[i] = 0.003
        elif j in z_cen_west:
            y_cen_west[i] = 0.003

    if plot == 'predicted':
        k.plot(x_cen/60,y_cen_east,color='orangered',label='Central Eastbound')
        k.plot(x_cen/60,y_cen_west,color='red',label='Central Westbound')

    #judged times - central 
    judged_cen_east = [(188,210),(498,516),(806,824),(1463,1481)]
    judged_cen_west = [(87,100),(303,319),(650,677),(904,925),(1307,1320),(1494,1506)]
    if 'cen' in lines and plot == 'judged':
        for start,end in judged_cen_east:
            k.axvspan(start/60, end/60, color='orange', alpha=0.4)
        for start,end in judged_cen_west:
            k.axvspan(start/60, end/60, color='red', alpha=0.4)

    #expected times - bakerloo
    expected_times_bak_southbound = [0,3.5,7,10,13.5,17,20,23.5,27,30,33.5,37,40,43.5,47,50,53.5,57,60,63.5,67,70]
    expected_times_bak_northbound = [1.5,4.5,8,11.5,14.5,18,21.5,24.5,28,31.5,34.5,38,41.5,44.5,48,51.5,54.5,58,61.5,64.5,68]
    z_bak_south, z_bak_north = find_bakerloo_line_passing_times(southbound=expected_times_bak_southbound, northbound=expected_times_bak_northbound, start_time=32851/3600)   
    x_bak = np.linspace(0,3600, 7201)
    y_bak_south = [0] * len(x_bak)
    y_bak_north = [0] * len(x_bak)
    for i, j in enumerate(x_bak):
        if j in z_bak_south:
            y_bak_south[i] = 0.003
        elif j in z_bak_north:
            y_bak_north[i] = 0.003
    
    if plot == 'predicted':
        k.plot(x_bak/60,y_bak_south,color='peru',label='Bakerloo Southbound')
        k.plot(x_bak/60,y_bak_north,color='sienna',label='Bakerloo Northbound')

    #judged times - bakerloo 
    judged_bak_south = [(286,302),(493,508),(752,780),(1055,1073),(1157,1173),(1510,1522)]
    judged_bak_north = [(350,370),(956,972),(1540,1562)]
    if 'bak' in lines and plot == 'judged':
        for start,end in judged_bak_south:
            k.axvspan(start/60, end/60, color='peru', alpha=0.4)
        for start,end in judged_bak_north:
            k.axvspan(start/60, end/60, color='sienna', alpha=0.4)

    #expected times - victoria
    expected_times_vic_southbound = [-.5,2,4.5,7,9.5,12,14.5,17,19.5,22,24.5,27,29.5,32,34.5,37,39.5,42,44.5,47,49.5,52,54.5,57,59.5,62,64.5,67,69.5]
    expected_times_vic_northbound = [-1,1.5,4,6.5,9,11.5,14,16.5,19,21.5,24,26.5,29,31.5,34,36.5,39,41.5,44,46.5,49,51.5,54,56.5,59,61.5,64,66.5,69]
    z_vic_south, z_vic_north = find_victoria_line_passing_times(southbound=expected_times_vic_southbound, northbound=expected_times_vic_northbound, start_time=32851/3600)   
    x_vic = np.linspace(0,3600, 7201)
    y_vic_south = [0] * len(x_vic)
    y_vic_north = [0] * len(x_vic)
    for i, j in enumerate(x_vic):
        if j in z_vic_south:
            y_vic_south[i] = 0.003
        elif j in z_vic_north:
            y_vic_north[i] = 0.003
    
    if plot == 'predicted':
        k.plot(x_vic/60,y_vic_south,color='dodgerblue',label='Victoria Southbound')
        k.plot(x_vic/60,y_vic_north,color='blue',label='Victoria Northbound')

    #judged times - victoria 
    judged_vic_south = [(140,175),(716,730),(1357,1373)]
    #judged_vic_north = [()]
    if 'vic' in lines and plot == 'judged':
        for start,end in judged_vic_south:
            k.axvspan(start/60, end/60, color='lightblue', alpha=0.4, label='Victoria Southbound')
        """for start,end in judged_vic_north:
            k.axvspan(start/60, end/60, color='blue', alpha=0.4)"""

def plot_hour01_trains(k, lines, plot):
    #expected times - central
    expected_times_cen_eastbound = [3,8,13,18,27.25]
    expected_times_cen_westbound = [5.5,10.5,15.5,20.5,26.5,30]
    z_cen_east, z_cen_west = find_central_line_passing_times(eastbound=expected_times_cen_eastbound, westbound=expected_times_cen_westbound, start_time=6.95)   
    x_cen = np.linspace(0,3600, 7201)
    y_cen_east = [0] * len(x_cen)
    y_cen_west = [0] * len(x_cen)
    for i, j in enumerate(x_cen):
        if j in z_cen_east:
            y_cen_east[i] = 0.004
        elif j in z_cen_west:
            y_cen_west[i] = 0.004
    
    if plot == 'predicted':
        k.plot(x_cen/60,y_cen_east,color='orangered',label='Central Eastbound')
        k.plot(x_cen/60,y_cen_west,color='red',label='Central Westbound')

    #judged times - central 
    judged_cen_east = np.array(pd.read_csv('judged_timings/cen_eas_01.csv').timings)
    judged_cen_west = np.array(pd.read_csv('judged_timings/cen_wes_01.csv').timings)
    
    if 'cen_eas' in lines and plot == 'judged':
        for line in judged_cen_east:
            k.axvline(line, color='orange', alpha=0.4)
    if 'cen_wes' in lines and plot == 'judged':
        for line in judged_cen_west:
            k.axvline(line, color='red', alpha=0.4)

    #expected times - bakerloo
    expected_times_bak_southbound = [1.5,5.5,10.5,15.5,20.5,25.5,27.5,31]
    expected_times_bak_northbound = [0,5.5,11.5,21.5,31.5]
    z_bak_south, z_bak_north = find_bakerloo_line_passing_times(southbound=expected_times_bak_southbound, northbound=expected_times_bak_northbound, start_time=417/60)   
    x_bak = np.linspace(0,3600, 7201)
    y_bak_south = [0] * len(x_bak)
    y_bak_north = [0] * len(x_bak)
    for i, j in enumerate(x_bak):
        if j in z_bak_south:
            y_bak_south[i] = 0.004
        elif j in z_bak_north:
            y_bak_north[i] = 0.004
    
    if plot == 'predicted':
        k.plot(x_bak/60,y_bak_south,color='peru',label='Bakerloo Southbound')
        k.plot(x_bak/60,y_bak_north,color='sienna',label='Bakerloo Northbound')

    #judged times - bakerloo 
    judged_bak_south = np.array(pd.read_csv('judged_timings/bak_sou_01.csv').timings)
    judged_bak_north = np.array(pd.read_csv('judged_timings/bak_nor_01.csv').timings)
    
    if 'bak_sou' in lines and plot == 'judged':
        for line in judged_bak_south:
            k.axvline(line, color='peru', alpha=0.4)
    if 'bak_nor' in lines and plot == 'judged':
        for line in judged_bak_north:
            k.axvline(line, color='sienna', alpha=0.4)

    #expected times - victoria
    expected_times_vic_southbound = [4.25,6.25,9.25,19.5,30]
    expected_times_vic_northbound = [2,4,7,12,17,22,24.25,27,32,37]
    z_vic_south, z_vic_north = find_victoria_line_passing_times(southbound=expected_times_vic_southbound, northbound=expected_times_vic_northbound, start_time=417/60)   
    x_vic = np.linspace(0,3600, 7201)
    y_vic_south = [0] * len(x_vic)
    y_vic_north = [0] * len(x_vic)
    for i, j in enumerate(x_vic):
        if j in z_vic_south:
            y_vic_south[i] = 0.004
        elif j in z_vic_north:
            y_vic_north[i] = 0.004
    
    if plot == 'predicted':
        k.plot(x_vic/60,y_vic_south,color='dodgerblue',label='Victoria Southbound')
        k.plot(x_vic/60,y_vic_north,color='blue',label='Victoria Northbound')

    #judged times - victoria 
    judged_vic_south = np.array(pd.read_csv('judged_timings/vic_sou_01.csv').timings)
    judged_vic_north = np.array(pd.read_csv('judged_timings/vic_nor_01.csv').timings)
    
    if 'vic_sou' in lines and plot == 'judged':
        for line in judged_vic_south:
            k.axvline(line, color='blue', alpha=0.4)
    if 'vic_nor' in lines and plot == 'judged':
        for line in judged_vic_north:
            k.axvline(line, color='dodgerblue', alpha=0.4)




