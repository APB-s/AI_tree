# some import
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Some information:
"""
CAR car acceptability

. PRICE overall price
. . buying buying price
. . maint price of the maintenance

. TECH technical characteristics
. . COMFORT comfort
. . . doors number of doors
. . . persons capacity in terms of persons to carry
. . . lug_boot the size of luggage boot
. . safety estimated safety of the car
"""

def read_data():
    # datasets column's name
    col_names = ['Buying', 'Maint', 'Nb_doors', 'Nb_place', 'Lug_capacity', 'Safety', 'Class_value']

    # read the data (stored locally)
    data_car = pd.read_csv('/home/apb/Cours/AI/Project_AI/AI_tree/car.csv', delimiter=",", names=col_names)
    return data_car

def traitement_donnee():
    data_car = read_data()
    # number of data in the datasets
    number_data = data_car.count()


    # replace string value into int
    data_car['Buying'] = data_car['Buying'].replace(['low', 'med', 'high', 'vhigh'], ['1', '2', '3', '4'])
    data_car['Maint'] = data_car['Maint'].replace(['low', 'med', 'high', 'vhigh'], ['1', '2', '3', '4'])
    data_car['Lug_capacity'] = data_car['Lug_capacity'].replace(['small', 'med', 'big'], ['1', '2', '3'])
    data_car['Safety'] = data_car['Safety'].replace(['low', 'med', 'high'], ['1', '2', '3'])

    return data_car

def decisiontree():
    from sklearn.tree import export_graphviz
    from six import StringIO
    from IPython.display import Image
    import pydotplus

    data_car = traitement_donnee()
    feature_cols = ['Buying', 'Maint', 'Nb_doors', 'Nb_place', 'Lug_capacity', 'Safety']
    X = data_car[feature_cols]
    y = data_car.Class_value

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=50)

    # data training
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    # prédiction faites à partir des données (30%)
    y_pred = clf.predict(X_test)

    # In our class, we did not talk about confusion matrix so I don't really think it is revellant
    # to talk about confusion_matrix..
    """
    result = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(result)
    """

    # Print the different result
    result1 = classification_report(y_test, y_pred)
    print("Classification Report:\n", )
    print(result1)
    result2 = accuracy_score(y_test, y_pred)
    print("Accuracy:", result2)

    value_toreturn = [clf, feature_cols]
    return value_toreturn

def tree_printing():
    from sklearn.tree import export_graphviz
    from six import StringIO
    from IPython.display import Image
    import pydotplus

    # important value from previous function
    clf = decisiontree()[0]
    feature_cols = decisiontree()[1]

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_cols, class_names=['Buying', 'Maint', 'Nb_doors', 'Nb_place', 'Lug_capacity', 'Safety', 'Class_value'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('value_car_tree.png')
    Image(graph.create_png())



decisiontree()
tree_printing()