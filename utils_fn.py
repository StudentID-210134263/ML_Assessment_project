## contains functions that are required for analyis of models to prevent duplicate
## code
import sklearn
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ClassificationAnalysis:
    """
    wrapper class over sklearn.metrics module to prevent duplicate code throughout the notebook.
    Requires the testing set of data as well as a pre-trained model from sklearn
    
    """
    def __init__(self, X_test: np.array, y_test: np.array, model: sklearn,
                 model_name: str):
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.model_name= model_name
    
    def model(self)-> sklearn:
        return self.model
    
    def visualise_confusion_matrix(self):
        """
        visualises a confusion matrix of a  pre-trained model in order to access performance of the model
        while also returning  metrics Sensitivity & Specificity

        Top Left Quadrant - TP, model correctly identifying case of diabetes
        Top Right Quadrant - FP, model incorrectly predicting case of diabetes for patient without diabetes
        Bottom Left Quadrant - FN, model incorrectly predicting no case of diabetes for patient with diabetes
        Bottomg Right Quadrant - TN, model correctly identfying no case of diabetes

        """
        confusion_matrix = metrics.confusion_matrix(self.y_test, self.model.predict(self.X_test))
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, 
                                                    display_labels = ['Diabetes', 'No Diabetes'])

        cm_display.plot()
        plt.title(f'confusion matrix for {self.model_name}')
        plt.show()


    
    def reciever_operator_curve(self) ->plt.show:
        """
        calculates an ROC curve as well as the AUC (Area Under Curve). 

        The closer the ROC curve is to the upper left corner of the graph, 
        the higher the accuracy of the test because in the upper left corner, 
        the sensitivity = 1 and the false positive rate = 0 (specificity = 1).

        A model that classifies test data perfectly will have an AUC of 1

        """

        pred_probability = self.model.predict(self.X_test)
        auc_score = metrics.roc_auc_score(self.y_test, pred_probability)
        print(f'area under curve for {self.model_name} is {auc_score}')

        fpr1, tpr1, thresh1 = metrics.roc_curve(self.y_test, pred_probability, pos_label=1)
        plt.plot(fpr1, tpr1, linestyle='--',color='black', label=self.model_name)
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        plt.show()
        
    def metrics(self)-> str:
        """
        calculates precision and recall
        
        precision - the quality of a positive prediction made by the model
        recall - the fraction of relevant instances that were retrieved.
        """
        
        precision = metrics.precision_score(self.y_test, self.model.predict(self.X_test))
        recall = metrics.recall_score(self.y_test, self.model.predict(self.X_test))
        print(metrics.classification_report(self.y_test, self.model.predict(self.X_test)))
        
        return f"""precision score {precision}  recall score {recall}"""
    
    def incorrect_model_probabilities(self)-> pd.DataFrame:
        """
        returns a dataframe of the data inputted in the model
        and only shows rows that the model has incorrectly classified
        The zeroth index is the liklihood that No diabetes occurs (0) and first
        index is the liklihood that diabetes does occur
        """
        
        actual = self.y_test.to_list()
        predictions = self.model.predict(self.X_test)
        
        incorrect_classified_indexes = [index for index in range(0, len(actual)) 
                                        if actual[index] != predictions[index]]
        
        X_test_reset_index = self.X_test.reset_index(drop=True)
        y_test_reset_index = self.y_test.reset_index(drop=True)
        
        
        info = X_test_reset_index.loc[incorrect_classified_indexes,
                                                'pregnancies':]
        info['actual_outcome'] = y_test_reset_index.loc[incorrect_classified_indexes]
        info['predicted_outcome'] = [predictions[index]
                                     for index in incorrect_classified_indexes]
        info['prediction_prob'] = [list(self.model.predict_proba(self.X_test)[index])
                                   for index in incorrect_classified_indexes]
        
        return info

def visualise_decision_trees(model):
    """Inspects samples of decision tree within a random forest model"""
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (50,50), dpi=300)
    tree.plot_tree(model)
    return plt.show()