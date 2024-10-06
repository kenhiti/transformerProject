import random
from keras import preprocessing as pp

def set_paddings(inputs, max_length):
    padded_list = pp.sequence.pad_sequences(inputs, value=0, padding='post', maxlen=max_length)
    __check_padding__(padded_list)
    return padded_list

def __check_padding__(str_list: list[str]):
    print(f'Print data to check after padding')
    for _ in range(5):
        print(str_list[random.randint(0, len(str_list) - 1)])