
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from skp4.classification_type import ClassificationType, get_classification_type

class Dataset:

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.classification_type: ClassificationType = get_classification_type(y_true, y_pred)
        
        if self.classification_type == ClassificationType.MULTI_CLASS:
            lb = LabelBinarizer()
            self.y_true_binarized = lb.fit_transform(self.y_true)
            self.y_pred_binarized = lb.transform(self.y_pred)

        elif self.classification_type  == ClassificationType.MULTI_LABEL:
            mlb = MultiLabelBinarizer()
            self.y_true_binarized = mlb.fit_transform(self.y_true)
            self.y_pred_binarized = mlb.transform(self.y_pred)


    def confusion_matrix(self):
        match self.classification_type:
            case ClassificationType.BINARY:
                return confusion_matrix(self.y_true, self.y_pred)
            case ClassificationType.MULTI_LABEL | ClassificationType.MULTI_CLASS:
                return multilabel_confusion_matrix(self.y_true_binarized, self.y_pred_binarized)
            

    def is_binary(self) -> bool:
        return self.classification_type == ClassificationType.BINARY