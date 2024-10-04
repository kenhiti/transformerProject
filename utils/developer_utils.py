def remove_long_sentences(input_list, output_list, max_length):
    idx_to_remove_inputs = [count for count, sent in enumerate(input_list) if len(sent) > max_length]
    __iterator_for_delete__(idx_to_remove_inputs, input_list, output_list)
    idx_to_remove_outputs = [count for count, sent in enumerate(output_list) if len(sent) > max_length]
    __iterator_for_delete__(idx_to_remove_outputs, input_list, output_list)

def __iterator_for_delete__(idx_to_remove, input_list, output_list):
    print(f'Input list length BEFORE to remove is:{len(input_list)} and output list length BEFORE to remove is: {len(output_list)}')
    for idx in reversed(idx_to_remove):
        del input_list[idx]
        del output_list[idx]
    print(f'Output list length AFTER to remove is:{len(input_list)} and output list length AFTER to remove is: {len(output_list)}')
