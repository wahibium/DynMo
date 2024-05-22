import datetime
import subprocess

def get_finish_time(filename):
    # 2023-01-12 02:48:39
    datetime_str = subprocess.check_output(['grep', 'after training is done', filename], text=True).split("datetime:")[-1].strip()
    datetime_object = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')

    return datetime_object

def get_start_time(filename):
    # 2023-01-12 02:48:39
    datetime_str = subprocess.check_output(['grep', 'before the start of training step', filename], text=True).split("datetime:")[-1].strip()
    datetime_object = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')

    return datetime_object

def get_diff(end_time, start_time):
    return (end_time - start_time).total_seconds()

if __name__ == "__main__":
    filenames = ["packing_24_8_8_2_64.txt", "packing_32_8_8_2_64.txt", "packing_40_8_8_2_64.txt", "packing_48_8_8_2_64.txt"]

    for filename in filenames:
	    start_time = get_start_time(filename)
	    finish_time = get_finish_time(filename)
	    diff = get_diff(finish_time, start_time)
	    print(f'{diff}')
