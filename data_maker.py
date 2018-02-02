from faker import Faker
import numpy as np
fake = Faker()
fake.seed_instance(42)
def data(n):
    array = []
    for _ in range(n):
        array.append(fake.name())
    return array


def apply_to_zeros(lst, dtype=np.int64):
    inner_max_len = max(map(len, lst))
    result = np.zeros([len(lst), inner_max_len], dtype)
    for i, row in enumerate(lst):
        for j, val in enumerate(row):
            result[i][j] = val
    return result

def convert_data(array):
    convert = []
    for i in range(len(array)):
        convert.append([ord(c.lower()) for c in array[i]])
    array =  np.asarray(convert).astype(np.float32)
    return apply_to_zeros(array)

print(convert_data(data(10)))
