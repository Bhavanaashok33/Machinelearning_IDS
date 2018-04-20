import pyspark
# import hoeffdingtree
import Utils
sc = pyspark.SparkContext(appName="HybridSparkModel")
from pyspark.mllib import classification
from pyspark.mllib import tree
from pyspark.mllib import regression
# from collection import defaultdict
# from hoeffdingtree import *

class HybridModel():
    def __init__(self, train, confidence_threshold=0.5):
        self.train = train
        self.test = []
        self.confident_data = 0          #count number of data points with confident prediction
        self.ambiguous_datacount = 0     #count number of data points with ambiguous prediction
        self.AccModel1 = 0.0
        self.AccModel2 = 0.0
        self.AccModel3 = 0.0
        self.ambiguous_data = []
        self.confidence_threshold = confidence_threshold
        # self.ambiguousTPs = [];
        # self.confidentTPs = [];
        # self.ambiguousTNs = [];
        # self.confidentTNs = [];
        # self.ambiguousFPs = [];
        # self.confidentFPs = [];
        # self.ambiguousFNs = [];
        # self.confidentFNs = [];
        self.maliciousSamples = [];
        self.benignSamples = [];
        # self.TNdict = {}
        # self.TPdict = {}
        # self.FNdict = {}
        # self.FPdict = {}
        self.TNarr = [[] for i in range(3)]       #[[],[],[]]
        self.TParr = [[] for j in range(3)]
        self.FNarr = [[] for k in range(3)]
        self.FParr = [[] for l in range(3)]
        self.TNRarr = ["" for a in range(3)]
        self.TPRarr = ["" for b in range(3)]
        self.FNRarr = ["" for c in range(3)]
        self.FPRarr = ["" for d in range(3)]
        self.PPVarr = ["" for e in range(3)]
        self.NPVarr = ["" for f in range(3)]
        self.FDRarr = ["" for g in range(3)]
        self.F1arr = ["" for h in range(3)]
            # defaultdict(dict)


        train_rdd = sc.parallelize(map(lambda x: regression.LabeledPoint(x[-1], x[:-1]), self.train))
        self.models = [
            # classification.LogisticRegressionWithSGD.train(data=train_rdd, iterations=10, step=1.0, miniBatchFraction=1.0, initialWeights=None, regParam=0.01, regType='l2', intercept=False, validateData=True, convergenceTol=0.001),
            # classification.LogisticRegressionWithLBFGS.train(data=train_rdd, iterations=10, initialWeights=None, regParam=0.01, regType='l2', intercept=False, corrections=10, tolerance=0.0001, validateData=True, numClasses=2),
            # classification.SVMWithSGD.train(data=train_rdd, iterations=10, step=1.0, regParam=0.01, miniBatchFraction=1.0, initialWeights=None, regType='l2', intercept=False, validateData=True, convergenceTol=0.001),
            # classification.NaiveBayes.train(data=train_rdd, lambda_=1.0),
            tree.DecisionTree.trainClassifier(data=train_rdd, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0),

            tree.RandomForest.trainClassifier(data=train_rdd, numClasses=2, categoricalFeaturesInfo={}, numTrees=3, featureSubsetStrategy='auto', impurity='gini', maxDepth=4, maxBins=32, seed=None),
            tree.GradientBoostedTrees.trainClassifier(data=train_rdd, categoricalFeaturesInfo={}, loss='logLoss', numIterations=10, learningRate=0.1, maxDepth=3, maxBins=32)
        ]

    def run_test(self, test):
        self.test = test
        for data_point in self.test:
            decisions = map(lambda model: int(model.predict(data_point[:-1])), self.models)   #Creates an array of decisions(Malicious/Benign) -> (1/0).

            prediction = Utils.mode(decisions)        #Higher the number of 1s => more likely to be Malicious than Benign.

            confidence = float(decisions.count(prediction)) / len(decisions) #calculates how confident the model is on the prediction. {# of Prediction Outcomes in Decision array / Length}

            print decisions, data_point[-1]                        #prediction, confidence, Status of processing data set,decisions=> array, prediction=>0/1.0, Confidence=>0.0 to 1, last data_point=> 1/0
            if confidence > self.confidence_threshold:              #compare confidence of the whole model with predefined confidence
                self.confident_data +=1                             #further classify each data point as confident or ambiguous
            else:
                self.ambiguous_datacount +=1
                self.ambiguous_data.append(data_point)              #Further process ambiguous data

            j=0
            for i in decisions:
                if decisions[i] == float(data_point[-1]):
                    if decisions[i] == 1:
                        self.TNarr[j].append(data_point)             #j=>data_point
                        j+=1                                #Logic working
                    else:
                        self.TParr[j].append(data_point)
                        j += 1
                else:
                    if decisions[i] == 1:
                        self.FNarr[j].append(data_point)
                        j += 1
                    else:
                        self.FParr[j].append(data_point)
                        j += 1
              # =========================Uncomment the above logic acc to convenience and check if it is working fine
            # print self.TNarr,self.FParr,self.FNarr,self.TParr
            print self.confident_data,self.ambiguous_datacount
            # print self.TNarr              #=====================check output outside decisions for loop
                        # if j==(len(decisions)-1):
                        #     j=0
    def accuracymodels(self):

        self.AccModel1 = (float(len(self.TParr[0])) + float(len(self.TNarr[0]))) / (float(len(self.TParr[0])) + float(len(self.TNarr[0])) + float(len(self.FParr[0])) + float(len(self.FNarr[0])))
        self.AccModel2 = (float(len(self.TParr[1])) + float(len(self.TNarr[1]))) / (float(len(self.TParr[1])) + float(len(self.TNarr[1])) + float(len(self.FParr[1])) + float(len(self.FNarr[1])))
        self.AccModel3 = (float(len(self.TParr[2])) + float(len(self.TNarr[2]))) / (float(len(self.TParr[2])) + float(len(self.TNarr[2])) + float(len(self.FParr[2])) + float(len(self.FNarr[2])))



    def Overall_Accuracy(self):         #Call from Main
        self.accuracymodels()
        return {
            "Overall_accuracy":(self.AccModel1 + self.AccModel2 + self.AccModel3)/3,
            "Accuracy_Model1" : self.AccModel1,
            "Accuracy_Model2": self.AccModel2,
            "Accuracy_Model3": self.AccModel3
        }

    def calc_performance(self):
        for j in range(3):
            self.TPRarr[j] = float(len(self.TParr[j])) / (float(len(self.TParr[j])) + float(len(self.FNarr[j])))
            self.TNRarr[j] = float(len(self.TNarr[j])) / (float(len(self.TNarr[j])) + float(len(self.FParr[j])))
            self.PPVarr[j] = float(len(self.TParr[j])) / (float(len(self.TParr[j])) + float(len(self.FParr[j])))
            self.NPVarr[j] = float(len(self.TNarr[j])) / (float(len(self.TNarr[j])) + float(len(self.FNarr[j])))
            self.FPRarr[j] = float(len(self.FParr[j])) / (float(len(self.FParr[j])) + float(len(self.TNarr[j])))
            self.FNRarr[j] = float(len(self.FNarr[j])) / (float(len(self.FNarr[j])) + float(len(self.TParr[j])))
            self.FDRarr[j] = float(len(self.FParr[j])) / (float(len(self.FParr[j])) + float(len(self.TParr[j])))
            self.F1arr[j] = (2 * float(len(self.TParr[j]))) / (2 * float(len(self.TParr[j])) + float(len(self.FParr[j])) + float(len(self.FNarr[j])))

    def Overall_Performance(self):          #Call from Main
        self.calc_performance()
        return {
            "TPR": (float(len(self.TParr[0])) + float(len(self.TParr[1])) + float(len(self.TParr[2])))/(float(len(self.TParr[0])) + float(len(self.TParr[1])) + float(len(self.TParr[2])) + float(len(self.FNarr[0])) + float(len(self.FNarr[1])) + float(len(self.FNarr[2]))),         #(float(self.TPRarr[0]) + float(self.TPRarr[1]) + float(self.TPRarr[2]))/3,
            "TNR": (float(len(self.TNarr[0])) + float(len(self.TNarr[1])) + float(len(self.TNarr[2])))/(float(len(self.TNarr[0])) + float(len(self.TNarr[1])) + float(len(self.TNarr[2])) + float(len(self.FParr[0])) + float(len(self.FParr[1])) + float(len(self.FParr[2]))),                 #(float(self.TNRarr[0]) + float(self.TNRarr[1]) + float(self.TNRarr[2]))/3,
            "FPR": (float(len(self.FParr[0])) + float(len(self.FParr[1])) + float(len(self.FParr[2])))/(float(len(self.FParr[0])) + float(len(self.FParr[1])) + float(len(self.FParr[2])) + float(len(self.TNarr[0])) + float(len(self.TNarr[1])) + float(len(self.TNarr[2]))),            #(float(self.FPRarr[0]) + float(self.FPRarr[1]) + float(self.FPRarr[2]))/3,
            "FNR": (float(len(self.FNarr[0])) + float(len(self.FNarr[1])) + float(len(self.FNarr[2])))/(float(len(self.FNarr[0])) + float(len(self.FNarr[1])) + float(len(self.FNarr[2])) + float(len(self.TParr[0])) + float(len(self.TParr[1])) + float(len(self.TParr[2]))),        #(float(self.FNRarr[0]) + float(self.FNRarr[1]) + float(self.FNRarr[2]))/3,
            "FDR": (float(len(self.FParr[0])) + float(len(self.FParr[1])) + float(len(self.FParr[2])))/(float(len(self.FParr[0])) + float(len(self.FParr[1])) + float(len(self.FParr[2])) + float(len(self.TParr[0])) + float(len(self.TParr[1])) + float(len(self.TParr[2]))),
            "TPR_Model1" : self.TPRarr[0],
            "TPR_Model2": self.TPRarr[1],
            "TPR_Model3": self.TPRarr[2],
            "TNR_Model1": self.TNRarr[0],
            "TNR_Model2": self.TNRarr[1],
            "TNR_Model3": self.TNRarr[2],
            "FPR_Model1": self.FPRarr[0],
            "FPR_Model2": self.FPRarr[1],
            "FPR_Model3": self.FPRarr[2],
            "FNR_Model1": self.FNRarr[0],
            "FNR_Model2": self.FNRarr[1],
            "FNR_Model3": self.FNRarr[2],
            "NPV_Model1": self.NPVarr[0],
            "NPV_Model2": self.NPVarr[1],
            "NPV_Model3": self.NPVarr[2],
            "PPV_Model1": self.PPVarr[0],
            "PPV_Model2": self.PPVarr[1],
            "PPV_Model3": self.PPVarr[2],
            "F1_Model1": self.F1arr[0],
            "F1_Model2": self.F1arr[1],
            "F1_Model3": self.F1arr[2],
            "FDR_Model1": self.FDRarr[0],
            "FDR_Model2": self.FDRarr[1],
            "FDR_Model3": self.FDRarr[2]
        }

    def confidence_score(self):             #Call from Main
        return {"confidence_score" : float(self.confident_data) / (float(self.confident_data) + float(self.ambiguous_datacount))}
    # {
        #     "TPR" : float(len(self.TPs())) / (float(len(self.TPs())) + float(len(self.FNs()))),
        #     "TNR" : float(len(self.TNs())) / (float(len(self.TNs())) + float(len(self.FPs()))),
        #     "PPV" : float(len(self.TPs())) / (float(len(self.TPs())) + float(len(self.FPs()))),
        #     "NPV" : float(len(self.TNs())) / (float(len(self.TNs())) + float(len(self.FNs()))),
        #     "FPR" : float(len(self.FPs())) / (float(len(self.FPs())) + float(len(self.TNs()))),
        #     "FNR" : float(len(self.FNs())) / (float(len(self.FNs())) + float(len(self.TPs()))),
        #     "FDR" : float(len(self.FPs())) / (float(len(self.FPs())) + float(len(self.TPs()))),
        #     "ACC" : (float(len(self.TPs())) + float(len(self.TNs()))) / (float(len(self.TPs())) + float(len(self.TNs())) + float(len(self.FPs())) + float(len(self.FNs()))),
        #     # "ACC": (float(len(self.confidentTPs)) + float(len(self.confidentTNs))) / (float(len(self.confidentTPs)) + float(len(self.confidentTNs)) + float(len(self.confidentFPs)) + float(len(self.confidentFNs))),
        #     "F1" : (2 * float(len(self.TPs()))) / (2 * float(len(self.TPs())) + float(len(self.FPs())) + float(len(self.FNs()))),
        #     "Confidence_score" : float(self.confident_data)/(float(self.confident_data) + float(self.ambiguous_data))
        # }
        #===================Figure out how to extract length of individual TP,TN,FP,FN and calculate accuracy for each [i]
        #===================Do not calculate any other parameter for now
                #     else:
                #         self.confidentTPs.append(data_point)
                # else:
                #     if decisions[i] == 1:
                #         self.confidentFNs.append(data_point)
                #     else:
                #         self.confidentFPs.append(data_point)

            # if confidence > self.confidence_threshold:
            #     if prediction == data_point[-1]:
            #         if prediction == 1:
            #             self.confidentTNs.append(data_point)
            #         else:
            #             self.confidentTPs.append(data_point)
            #     else:
            #         if prediction == 1:
            #             self.confidentFNs.append(data_point)
            #         else:
            #             self.confidentFPs.append(data_point)
            # else:
            #     if prediction == data_point[-1]:
            #         if prediction == 1:
            #             self.ambiguousTNs.append(data_point)
            #         else:
            #             self.ambiguousTPs.append(data_point)
            #     else:
            #         if prediction == 1:
            #             self.ambiguousFNs.append(data_point)
            #         else:
            #             self.ambiguousFPs.append(data_point)

    
    # def ambiguous(self):
    #     return self.ambiguousFNs + self.ambiguousFPs + self.ambiguousTNs + self.ambiguousTPs

    # def confident(self):
    #     return self.confidentFNs + self.confidentFPs + self.confidentTNs + self.confidentTPs
    #
    # def TPs(self):
    #     return self.confidentTPs        #self.ambiguousTPs +
    #
    # def TNs(self):
    #     return self.confidentTNs        #self.ambiguousTNs +
    #
    # def FPs(self):
    #     return self.confidentTPs        #self.ambiguousFPs +
    #
    # def FNs(self):
    #     return self.confidentFNs        #self.ambiguousFNs +
    #
    # def malicious(self):
    #     return self.TPs() + self.FNs()
    #
    # def benign(self):
    #     return self.FPs() + self.TNs()
    #
    # def accuracy(self):
    #     return {
    #         "TPR" : float(len(self.TPs())) / (float(len(self.TPs())) + float(len(self.FNs()))),
    #         "TNR" : float(len(self.TNs())) / (float(len(self.TNs())) + float(len(self.FPs()))),
    #         "PPV" : float(len(self.TPs())) / (float(len(self.TPs())) + float(len(self.FPs()))),
    #         "NPV" : float(len(self.TNs())) / (float(len(self.TNs())) + float(len(self.FNs()))),
    #         "FPR" : float(len(self.FPs())) / (float(len(self.FPs())) + float(len(self.TNs()))),
    #         "FNR" : float(len(self.FNs())) / (float(len(self.FNs())) + float(len(self.TPs()))),
    #         "FDR" : float(len(self.FPs())) / (float(len(self.FPs())) + float(len(self.TPs()))),
    #         "ACC" : (float(len(self.TPs())) + float(len(self.TNs()))) / (float(len(self.TPs())) + float(len(self.TNs())) + float(len(self.FPs())) + float(len(self.FNs()))),
    #         # "ACC": (float(len(self.confidentTPs)) + float(len(self.confidentTNs))) / (float(len(self.confidentTPs)) + float(len(self.confidentTNs)) + float(len(self.confidentFPs)) + float(len(self.confidentFNs))),
    #         "F1" : (2 * float(len(self.TPs()))) / (2 * float(len(self.TPs())) + float(len(self.FPs())) + float(len(self.FNs()))),
    #         "Confidence_score" : float(self.confident_data)/(float(self.confident_data) + float(self.ambiguous_data))
    #     }
    def parameters_used(self):          #call from Main
        return {
            "TP_Model1": float(len(self.TParr[0])),
            "TP_Model2": float(len(self.TParr[1])),
            "TP_Model3": float(len(self.TParr[2])),
            "TN_Model1": float(len(self.TNarr[0])),
            "TN_Model2": float(len(self.TNarr[1])),
            "TN_Model3": float(len(self.TNarr[2])),
            "FP_Model1": float(len(self.FParr[0])),
            "FP_Model2": float(len(self.FParr[1])),
            "FP_Model3": float(len(self.FParr[2])),
            "FN_Model1": float(len(self.FNarr[0])),
            "FN_Model2": float(len(self.FNarr[1])),
            "FN_Model3": float(len(self.FNarr[2])),
            "TPs": float(len(self.TParr[0])) + float(len(self.TParr[1])) + float(len(self.TParr[2])),
            "FNs": float(len(self.FNarr[0])) + float(len(self.FNarr[1])) + float(len(self.FNarr[2])),
            "FPs": float(len(self.FParr[0])) + float(len(self.FParr[1])) + float(len(self.FParr[2])),
            "TNs": float(len(self.TNarr[0])) + float(len(self.TNarr[1])) + float(len(self.TNarr[2]))
        }
