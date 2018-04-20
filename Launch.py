import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import findspark
findspark.init()
from pyspark import SparkContext
from pyspark import SparkConf
import Utils
# import pandas
# from sklearn import model_selection
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import VotingClassifier
from HybridModel import HybridModel

def main():
    with open("input.txt", "r+") as raw_data:
        data = map(lambda x: x.replace("\n", "").split("\t"), list(raw_data))
        # data = map(lambda x: [x[:17] + x[22:] + x[17] if (x[17] == '1') else 0], data)
        # if ((data[17] == '1') or (data[17] == '-1')):
        #     data = data[:17] + data[22:] + data[17]
        # else:
        #     del data
        #print data
        # print data
        # or (x[17] == '-1')
        data = map(lambda x: x[:14] + x[19:20] + x[21:] + [x[17] if x[17] == '1' else 0], data)
        data = map(Utils.vectorize, data)
        test, train = Utils.randomSplit([0.5, 0.5], data)
        model =  HybridModel(train)
        model.run_test(test)
        print model.Overall_Accuracy()
        print model.Overall_Performance()
        print model.confidence_score()
        # print model.accuracy()
        print model.parameters_used()



if __name__ == '__main__':
    main()

