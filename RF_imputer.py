import numpy as np
from decision_tree import *
from itertools import product

def one_hot_encode(X, attributes):
    new_num_att = 0
    for i in range(X.shape[1]):
        if i in attributes: new_num_att += len(np.unique(X[:, i]))
        else: new_num_att += 1

    X_new = np.zeros((X.shape[0], new_num_att))
    at_col = 0
    for i in range(X.shape[1]):
        if i in attributes:
            unique_val = np.unique(X[:, i])
            for j in range(len(unique_val)):
                X_new[:, at_col] = (X[:, i] == unique_val[j]).astype(float)
                at_col += 1
        else:
            X_new[:, at_col] = X[:, i]
            at_col += 1
    return X_new

def return_missing_postions(X, target_columns, placeholder):
    missing_positions = []
    for column in target_columns:
        col = X[:, column]
        for row in range(len(col)):
            if col[row] == placeholder:
                missing_positions.append((row, column))
    return missing_positions

def initial_guess(X, y, attribute_types, missing_positions, placeholder):
    X_new = X.copy()
    for pos in missing_positions:
        entries_of_interest = X[:, pos[1]][np.where((y==y[pos[0]]))[0]]
        entries_of_interest = entries_of_interest[entries_of_interest != placeholder]
        if attribute_types[pos[1]] == 'D': X_new[pos] = np.argmax(np.bincount(entries_of_interest))
        elif attribute_types[pos[1]] == 'C': X_new[pos] = np.median(entries_of_interest)
    return X_new

def update_proximity_matrix_single_tree(RF, block, proximity_matrix, node_indices=None):
    if node_indices is None: node_indices = np.array(range(RF.X.shape[0]))
    if block.type == 'Node':
        left_indices, right_indices = RF.make_split(RF.X, node_indices, block.feature, block.thresh)

        update_proximity_matrix_single_tree(RF, block.left_branch, proximity_matrix, left_indices)
        update_proximity_matrix_single_tree(RF, block.right_branch, proximity_matrix, right_indices)
    elif block.type == 'Leaf':
        update_positions = np.column_stack(list(product(node_indices, node_indices))).astype(int)
        proximity_matrix[update_positions[0], update_positions[1]] += 1

def get_proximity_matrix(RF):
    ensemble = RF.ensemble
    proximity_matrix = np.zeros((RF.X.shape[0], RF.X.shape[0]))
    for key in ensemble:
        update_proximity_matrix_single_tree(RF, ensemble[key], proximity_matrix)
    np.fill_diagonal(proximity_matrix, 0) #set all the diagonal positions to zero
    proximity_matrix = proximity_matrix/len(ensemble)
    return proximity_matrix

def update_data(X, proximity_matrix, missing_positions, attribute_types, placeholder):
    X_new = X.copy()
    for pos in missing_positions:
        column = X[:, pos[1]]
        if attribute_types[pos[1]] == 'D':
            unique_vals, counts = np.unique(column, return_counts=True)
            unique_vals, counts = unique_vals[unique_vals!=placeholder], counts[unique_vals!=placeholder]
            max_weighted_freq = 0
            value_to_assign = X[pos]
            for i in range(len(unique_vals)):
                frequency = counts[i]/len(column[column!=placeholder])
                weight = np.sum(proximity_matrix[pos[0], np.where((column==unique_vals[i]))[0]])/(np.sum(proximity_matrix[pos[0]])+1e-08)
                weighted_frequency = frequency*weight
                if weighted_frequency > max_weighted_freq:# and unique_vals[i] != placeholder: ##########
                    max_weighted_freq = weighted_frequency
                    value_to_assign = unique_vals[i]
            X_new[pos] = value_to_assign

        elif attribute_types[pos[1]] == 'C':
            weights = proximity_matrix[pos[0]]/(np.sum(proximity_matrix[pos[0]]) + 1e-08)
            column[column==placeholder] = 0 ##########
            X_new[pos] = np.dot(column, weights)
    return X_new
    
def RF_imputer(X, y, placeholder, attribute_types, target_columns, iterations, 
               num_trees, num_features, max_depth):
    # attribute_types (dict) --> type of all the attributes (D/C)
    # target_columns (list) --> list of columns that have missing values
    # placeholder --> number 
    discrete_columns = [key for key, value in attribute_types.items() if value == 'D']
    missing_positions = return_missing_postions(X, target_columns, placeholder)
    X_new = initial_guess(X, y, attribute_types, missing_positions, placeholder)
    
    for iteration in range(iterations):
        print(f"At iteration: {iteration+1}")
        X_one_hot = one_hot_encode(X_new, discrete_columns)
        RF = RandomForest(X_one_hot, y, 'classification')
        RF.build_ensemble(num_trees=num_trees, num_features=num_features, max_depth=max_depth, print_opt=False)
        proximity_matrix = get_proximity_matrix(RF)
        #print(proximity_matrix[proximity_matrix != 0])
        X_new = update_data(X, proximity_matrix, missing_positions, attribute_types, placeholder)

    return X_new

