"""An implementation of the Intersection over Union (IoU) metric for Keras."""
from keras import backend as K
import pandas as pd

def bb_iou(y_true, y_pred):
    trainPath = "/home/harshit1201/Desktop/Project:TrueSight/training_set.csv"
    cols =["image_name","x1","x2","y1","y2"]
    df2 = pd.read_csv(trainPath, skiprows=[0], header=None, names=cols)
    maxX1 = df2["x1"].max()
    maxX2 = df2["x2"].max()
    maxY1 = df2["y1"].max()
    maxY2 = df2["y2"].max()
    y_pred[0] = y_pred[0].flatten()
    y_pred[1] = y_pred[1].flatten()
    y_pred[2] = y_pred[2].flatten()
    y_pred[3] = y_pred[3].flatten()
    y_true[0]=y_true[0]*maxX1
    y_true[1]=y_true[1]*maxX2
    y_true[2]=y_true[2]*maxY1
    y_true[3]=y_true[3]*maxY2
    y_pred[0]=y_pred[0]*maxX1
    y_pred[1]=y_pred[1]*maxX2
    y_pred[2]=y_pred[2]*maxY1
    y_pred[3]=y_pred[3]*maxY2

    xA = max(y_true[0], y_pred[0])
    yA = max(y_true[1], y_pred[1])
    xB = min(y_true[2], y_pred[2])
    yB = min(y_true[3], y_pred[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    y_trueArea = (y_true[2] - y_true[0] + 1) * (y_true[3] - y_true[1] + 1)
    y_predArea = (y_pred[2] - y_pred[0] + 1) * (y_pred[3] - y_pred[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(y_trueArea + y_predArea - interArea)

    # return the intersection over union value
    return iou


def iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def build_iou_for(label: int, name: str=None):
    """
    Build an Intersection over Union (IoU) metric for a label.
    Args:
        label: the label to build the IoU metric for
        name: an optional name for debugging the built method
    Returns:
        a keras metric to evaluate IoU for the given label

    Note:
        label and name support list inputs for multiple labels
    """
    # handle recursive inputs (e.g. a list of labels and names)
    if isinstance(label, list):
        if isinstance(name, list):
            return [build_iou_for(l, n) for (l, n) in zip(label, name)]
        return [build_iou_for(l) for l in label]

    # build the method for returning the IoU of the given label
    def label_iou(y_true, y_pred):
        """
        Return the Intersection over Union (IoU) score for {0}.
        Args:
            y_true: the expected y values as a one-hot
            y_pred: the predicted y values as a one-hot or softmax output
        Returns:
            the scalar IoU value for the given label ({0})
        """.format(label)
        return iou(y_true, y_pred, label)

    # if no name is provided, us the label
    if name is None:
        name = label
    # change the name of the method for debugging
    label_iou.__name__ = 'iou_{}'.format(name)

    return label_iou


def mean_iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1]
    # initialize a variable to store total IoU in
    total_iou = K.variable(0)
    # iterate over labels to calculate IoU for
    for label in range(num_labels):
        total_iou = total_iou + iou(y_true, y_pred, label)
    # divide total IoU by number of labels to get mean IoU
    return total_iou / num_labels
