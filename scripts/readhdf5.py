import h5py
import argparse
import numpy as np

def normalize(array, min_soll, max_soll):
    normalized = []
    dif_soll = max_soll - min_soll
    dif_cur = np.max(array) - np.min(array)
    for value in array:
        tmp = (((value - np.min(array)) * dif_soll) / dif_cur) + min_soll
        normalized.append(tmp)
    return normalized

def extract_depth(filename):
    hfile = h5py.File(filename, "r+")
    print(*[item for item in hfile.items()], sep="\n")
    data = hfile['colors'][()]
    print(data)
    print("##########\n")
    ddata = hfile['depth'][()]
    print(ddata)
    print("##########\n")
    print(np.min(ddata))
    print(np.max(ddata))
    del hfile['depth']
    hfile.create_dataset('depth', data=normalize(ddata, 0, 1))
    hfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract depth data from image',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filename', '-f', help='location of hdf5-file', required=True)
    parser.add_argument('--normalize', '-n', help='normalize depth')
    args = parser.parse_args()

    extract_depth(args.filename)
