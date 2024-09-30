import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []

    field_types = [
        'int', 'float', 'int', 'float', 'int', 'float', 'float', 'float', 'float', 'float',
        'month', 'int', 'int', 'int', 'int', 'visitor', 'bool'
    ]
    months = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5, 'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}

    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)

        for row in csvreader:
            evidence_row = [convert_field(row[i], field_types[i]) for i in range(len(row) - 1)]
            evidence_row[10] = months[row[10]]
            label = convert_field(row[-1], "bool")
            evidence.append(evidence_row)
            labels.append(label)

        return (evidence, labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(evidence, labels)

    return knn

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Sensitivity is the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    Specificity is the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # Initialize counters
    true_positive = 0
    true_negative = 0
    total_positive = 0
    total_negative = 0

    # Iterate through labels and predictions once
    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            total_positive += 1
            if predicted == 1:
                true_positive += 1
        else:
            total_negative += 1
            if predicted == 0:
                true_negative += 1

    # Calculate sensitivity (true positive rate)
    sensitivity = true_positive / total_positive if total_positive != 0 else 0
    
    # Calculate specificity (true negative rate)
    specificity = true_negative / total_negative if total_negative != 0 else 0

    return (sensitivity, specificity)


def convert_field(value, field_type):
    if field_type == 'int':
        return int(value)
    elif field_type == 'float':
        return float(value)
    elif field_type == 'month':
        return 0
    elif field_type == 'visitor':
        return 1 if value == 'Returning_Visitor' else 0
    elif field_type == 'bool':
        return 1 if value == 'TRUE' else 0

if __name__ == "__main__":
    main()
