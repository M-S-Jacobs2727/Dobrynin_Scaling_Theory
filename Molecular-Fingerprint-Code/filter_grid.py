import os
import random
import numpy as np

random.seed(1)

def select_from_files(path):

    mylist = os.listdir(path)
    filter_groups = list(np.unique(random.sample(mylist, int(len(mylist)/2))))
    return filter_groups

def filter_grid(file_list, path, save_path):

    for i in file_list:
        data = np.genfromtxt(f"{path}{i}")
        data[:,1:] = 0
        np.savetxt(f"{save_path}{i}",data)


def main():

    path = "grid_data\\"
    save_path = "grid_filtered_data\\"
    filter_groups = select_from_files(path)
    filter_grid(filter_groups, path, save_path)

if __name__ == '__main__':
    main()