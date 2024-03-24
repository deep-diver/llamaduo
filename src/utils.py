def _sort_step_order(input_steps):
    correct_order = ["ft", "eval", "synth-gen", "deploy"]

    def custom_sort(input_steps):
        def get_index(value):
            try:
                return correct_order.index(value)
            except ValueError:  # In case a value is not in the correct order
                return float('inf')  # Assign a very high index to put it at the end

        input_steps.sort(key=get_index)
        return input_steps

    return custom_sort(input_steps)

def _all_steps_allowed(input_steps):
    allowed_steps = {"ft", "eval", "synth-gen", "deploy"}
    return all(value in allowed_steps for value in input_steps)

def _remove_duplicates(input_steps):
    return list(dict.fromkeys(input_steps))

def get_steps_right(input_steps):
    input_steps = _remove_duplicates(input_steps)
    valid = _all_steps_allowed(input_steps)
    if valid:
        input_steps = _sort_step_order(input_steps)
    else:
        input_steps = None
    
    return valid, input_steps