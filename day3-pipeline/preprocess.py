from sklearn.model_selection import train_test_split

def preprocess(data):
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test
