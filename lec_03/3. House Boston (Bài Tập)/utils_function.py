"""
AI Class

Utility functions

"""

def load_Boston_housing_data(test_ratio=0.2, feature_ind = None, random_state=0, print_info=False):
    """
    Load Boston Dataset from Sklearn.
    
    Args:
      test_ratio(float)       : a proportion between train set and test set. Default = 0.2
      feature_ind(list(int))  : a list of index feature to be extracted from the original data set. No indication
                                means that all feature is chosen.
      random_state(int)       : a seed value for shuffling between values
      print_info(boolean)     : True if print the information of data set. Default: False
      
    Returns:
      a tuple of four np.array data sets train_data, test_data, train_targets, test_targets
      
    """
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    boston_data = load_boston()
    
    if print_info:
        print(boston_data['DESCR'])
        
    if len(feature_ind) > 0:
        print("Selected original features are %s" % boston_data.feature_names[feature_ind])
        return train_test_split(boston_data.data[:, feature_ind], \
                                boston_data.target, \
                                test_size=test_ratio, \
                                random_state=random_state)
    else:
        print("All original features are selected")
        return train_test_split(boston_data.data, \
                                boston_data.target, \
                                test_size=test_ratio, \
                                random_state=random_state)

def timing(function, *args):
    """
    A function measured an amount of time to run a program
    """
    import time

    start = time.time()
    function(*args)
    end = time.time()
    return start, end

def scatter_plot(X, y, title="Scatter Plot", x_label="Name of feature", y_label="Name of targets"):
    """
    A function to draw a scatter plot 
    
    Args:
      X(np.array)      : Instances/ Examples/ Features/ Data
      y(np.array)      : Targets
      title (str)      : Title of a plot
      x_label (str)    : Name of x label
      y_label (str)    : Name of y label
      
    Returns:
      an visualization exported as an PNG image.
    """
    import matplotlib.pyplot as plt

    plt.title(title)
    plt.scatter(X, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    plt.close()

from seaborn import pairplot


