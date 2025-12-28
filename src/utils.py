#########################
#### CardiCat ###########
#########################


def flatten_list(list_of_lists):
    """Flattens a list of lists.
       Given a list of lists, returns a flatten list.

    Args:
        list_of_lists (list[lists]): A list of lists to be flattened
    Return:
        list: a flattened list
    """
    return [item for sublist in list_of_lists for item in sublist]
