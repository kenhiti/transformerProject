import re

def load_data(file_name: str) -> str:
    with open(file_name, mode='r', encoding='utf-8') as f:
        return f.read()
load_data.__doc__ = "A simple function to load data from file"

def split_data(data: str) -> list[str]:
    return data.split('\n')
split_data.__doc__ = "A simple function to split data from file"

def compare_length(data1: list, data2: list) -> bool:
    print(f'The first data size is {len(data1)} and the second data size is {len(data2)}')
    return len(data1) == len(data2)
compare_length.__doc__ = "A simple function to compare length of data"

def print_data_to_check_it(*args: list[str], log_it: bool):
    if log_it == True and len(args) == 2:
        data1, data2 = args

        for line in range(5):
            print('-----')
            print(data1[line])
            print(data2[line])
print_data_to_check_it.__doc__ = "A simple function to compare two lists of the strings and print this one "

def clean_data(*args: str) -> tuple[list[str], list[str]]:
    if len(args) == 2:
        data1, data2 = args
        data1 = __remove__unwanted_characters__(data1)
        data2 = __remove__unwanted_characters__(data2)
        print_data_to_check_it(data1, data2, log_it=False)
        return data1, data2
clean_data.__doc__ = "A function to prepare dataset. This one removes dots at the start of the phrases and double spaces. It's necessary because an algorithm can't interpret this as the ends of the sentence."

def check_load_dataset(data1: str, data2: str, log_it: bool):
    english_data = split_data(data1)
    portuguese_data = split_data(data2)
    print(f"Are these length datasets equal??? {compare_length(english_data, portuguese_data)}")
    print_data_to_check_it(english_data, portuguese_data, log_it =log_it)
check_load_dataset.__doc__ = "A simple function to check whether a dataset was loaded correctly"

def __remove__unwanted_characters__(data: str) -> list[str]:
    data = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", "###", data)
    data = re.sub(r"###", "", data)
    data = re.sub(r" +", " ", data)
    return split_data(data)
__remove__unwanted_characters__.__doc__ = "A private function to remove unwanted characters from a string"
