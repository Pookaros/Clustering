def truncate(n:float, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

# returns the unique elements of a list in a list
def unique_elements(list:list) -> list:
    new_list = []
    for element in list:
        if element not in new_list: new_list.append(element)
    return new_list

# returs the number of inique elements in a list
def unique_count(list:list) -> int:
    return len(unique_elements(list))