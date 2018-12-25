import numpy as np
import random

work_modes = {0: (3, 7),
              1: (12.5, 14.5),
              2: (20, 22),
              3: (26, 30)}


def generate_pulse(start_time, stop_time, height=1):
    time_list = [start_time, start_time+0.01, stop_time, stop_time+0.01]
    height = max(height, 0.55)
    values_list = [0, height, height, 0]
    return time_list, values_list


def generate_ramp(start_time, stop_time, start_height=0.55, diff=0.45):
    time_list = [start_time, start_time + 0.01, stop_time, stop_time + 0.01]
    start_height = max(start_height, 0.55)
    end_height = min(start_height+diff, 1)
    values_list = [0, start_height, end_height, 0]
    return time_list, values_list


def generate_stairs(start_time, stop_time, start_height=0.55, diff=0.45, num_of_stairs=3):
    times = np.linspace(start_time, stop_time, num_of_stairs+1)
    times = np.round(times, 2)
    go_up = times + 0.01
    time_list = sum([[x, y] for x, y in zip(times, go_up)], [])
    start_height = max(start_height, 0.55)
    end_height = min(start_height + diff, 1)
    values = np.linspace(start_height, end_height, num_of_stairs)
    values = np.round(values, 2)
    values_list = sum([[x, y] for x, y in zip(values, values)], [])
    values_list.insert(0, 0)
    values_list.append(0)
    return time_list, values_list


def generate_signal(work_mode_num, mode, my_time, my_val):
    possible_len = np.linspace(1.0, 2.0, 9)
    possible_len = np.round(possible_len, 2)

    random.shuffle(possible_len)
    event_len = possible_len[0]
    time_start = work_modes[work_mode_num][0]
    time_end = work_modes[work_mode_num][1]
    time_start_last = time_end - event_len
    possible_start = np.linspace(time_start, time_start_last, 9)
    possible_start = np.round(possible_start, 2)
    random.shuffle(possible_start)
    start = possible_start[0]
    end = start+event_len
    if mode == 'pulse':
        tim, val = generate_pulse(start, end)
    elif mode == 'ramp':
        tim, val = generate_ramp(start, end)
    elif mode == 'stairs':
        tim, val =  generate_stairs(start,end)

    my_time += tim
    my_val += val


time1 = [0]
time2 = [0]
time3 = [0]
value1 = [0]
value2 = [0]
value3 = [0]
generate_signal(0, 'pulse', time3, value3)
generate_signal(2, 'ramp', time1, value1)
generate_signal(1, 'stairs', time3, value3)
generate_signal(3, 'pulse', time2, value2)

time1.append(30)
time2.append(30)
time3.append(30)
value1.append(0)
value2.append(0)
value3.append(0)

with open('/home/kacper/Pulpit/your_file.txt', 'w') as f:
    f.writelines(["%s; " % item for item in time1])
    f.writelines(['\n'])
    f.writelines(["%s; " % item for item in value1])
    f.writelines(['\n'])
    f.writelines(["%s; " % item for item in time2])
    f.writelines(['\n'])
    f.writelines(["%s; " % item for item in value2])
    f.writelines(['\n'])
    f.writelines(["%s; " % item for item in time3])
    f.writelines(['\n'])
    f.writelines(["%s; " % item for item in value3])
    f.writelines(['\n'])

