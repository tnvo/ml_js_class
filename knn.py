# K-nearest Neighbor
"""
What is this person's job satisfaction right now?
1-5 Scale -> Output = 4

"""
def run_knn():
    import pandas as pd
    import numpy as np
    # to get the knn
    # You have nearest scattered data points and cluster them together
    from sklearn.neighbors import KNeighborsClassifier


    # read in dataset; there is a header
    dataset = pd.read_csv('Classification.csv', header=0)

    # take certain columns only
    columns = ['JobSatisfaction', 'Age', 'DistanceFromHome', 'YearsInCurrentRole']

    # create dataframe with just those columns
    df = dataset[columns]

    # clean up NA data with 0
    df.fillna(0, inplace=True)

    # test data = try to predict to see accuracy measurement off of that
    # 1 test data point

    # Split data into X axis and Y axis
    # get data on everything but first row bc that's header row
    # : = all the row; 1: = age to current row
    X = np.array(df.iloc[:, 1:])

    # Do the same for y - but for the JobSatisfaction Column
    y = np.array(df['JobSatisfaction'])


    # Build the model; Give it a number
    model = KNeighborsClassifier(n_neighbors=3)

    # Fit the model with the data -> Fit X and Y
    model.fit(X, y)


    # After you fit it, you want to have some type of validation and see if it work or not
    # Multidimensional array and matrix
    X_test = [[41, 1, 4]]

    # Get our predicted value from the model
    predicted = model.predict(X_test)

    print(predicted)
    return predicted
