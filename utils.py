def _sort_step_order(input_steps):
    """
    _sort_step_order makes sure the given input_steps to be organized in
    ["fine-tuning", "eval", "synth-gen", "deploy"] order
    """
    correct_order = ["fine-tuning", "batch-infer", "eval", "synth-gen", "deploy"]

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
    """
    _all_steps_allowed makes sure there are only allowed values in the input_steps
    """
    allowed_steps = {"fine-tuning", "batch-infer", "eval", "synth-gen", "deploy"}
    return all(value in allowed_steps for value in input_steps)

def _remove_duplicates(input_steps):
    """
    _remove_duplicates removes any duplicate values in the given input_steps
    """
    return list(dict.fromkeys(input_steps))

def validate_steps(input_steps):
    """
    get_steps_right on the given input_steps:
    1. remove duplicate values
    2. validation check if only allowed values are included
    3. makes sure the values are sorted in a certain order
    """
    input_steps = _remove_duplicates(input_steps)
    valid = _all_steps_allowed(input_steps)
    if valid:
        input_steps = _sort_step_order(input_steps)
    else:
        input_steps = None
    
    return valid, input_steps