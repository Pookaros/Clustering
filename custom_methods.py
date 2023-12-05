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

def get_cluster_no_from_dendro(dendrogram):
    ### custom 
    branches_heights = []
    branches_width_height = []
    res = []

    # It contains the list of (x_coord_left, y_coord, y_coord, x_coord_right) of the branch
    branches_width_height = [[round(dendrogram['dcoord'][i][0], 3),
                            round(dendrogram['dcoord'][i][1], 3),
                            round(dendrogram['dcoord'][i][2], 3),
                            round(dendrogram['dcoord'][i][3], 3)] for i in range(1, len(dendrogram['dcoord']))]

    # It contains only the height of eachs branch
    branches_heights = [element[1] for element in branches_width_height]

    # Sorted
    branches_heights_sorted = (sorted(branches_heights))

    # Selector is a list that contains the differences between the ordered heights
    selector = [branches_heights_sorted[i+1] - branches_heights_sorted[i] for i in range(len(branches_heights) - 1)]
    print(selector)
    # and we take the maximum of them
    temp = max(selector)

    # determine the list of the thresholds for making the clusters
    ress = [i for i, j in enumerate(selector) if j == temp]


    for j in range(0, len(ress)):
        flag = 0
        for i in range(0, len(branches_width_height)):

            if (branches_width_height[i][0] < branches_heights_sorted[ress[j]]*3/2 and branches_width_height[i][1] > branches_heights_sorted[ress[j]]*3/2) and not(branches_width_height[i][2] > branches_heights_sorted[ress[j]]*3/2 and branches_width_height[i][3] < branches_heights_sorted[ress[j]]*3/2):
                flag = flag+1

            elif (branches_width_height[i][2] > branches_heights_sorted[ress[j]]*3/2 and branches_width_height[i][3] < branches_heights_sorted[ress[j]]*3/2) and not(branches_width_height[i][0] < branches_heights_sorted[ress[j]]*3/2 and branches_width_height[i][1] > branches_heights_sorted[ress[j]]*3/2):
                flag = flag+1

            elif (branches_width_height[i][2] > branches_heights_sorted[ress[j]]*3/2 and branches_width_height[i][3] < branches_heights_sorted[ress[j]]*3/2) and (branches_width_height[i][0] < branches_heights_sorted[ress[j]]*3/2 and branches_width_height[i][1] > branches_heights_sorted[ress[j]]*3/2):
                flag = flag+2
        res.append(flag)
        
    return res