import cPickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
import numpy as np
import os
from os.path import exists
from os import makedirs
from sklearn import svm
from sklearn.metrics import confusion_matrix,accuracy_score,average_precision_score,f1_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model

from utils import get_all_files

def base_model(trainfile, model_dir, resultfile):

    # res_dir = 'arff/all_sensor/results/'
    # trainfile = dirname+filename
    # testfile = dirname + 'data_test.csv'
    data = np.genfromtxt(trainfile, delimiter=',')
    row, col = data.shape
    train_features = data[:, 0:col - 1]
    train_labels = data[:, col - 1]
    X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size = 0.2, random_state = 0)



    # svm train
    # clf_svm = Pipeline([
    #     ('feature_selection', SelectFromModel(LinearSVC())),
    #     ('classification', SVC())
    # ])
    # clf.fit(X, y)
    clf_svm = svm.SVC(decision_function_shape='ovo')
    clf_svm.fit(X_train, y_train)

    # linear svm
    # clf_linear_svm = Pipeline([
    #     ('feature_selection', SelectFromModel(LinearSVC())),
    #     ('classification', svm.LinearSVC())
    # ])
    clf_linear_svm = svm.LinearSVC()
    clf_linear_svm.fit(X_train, y_train)

    # decision tree
    # clf_dt = Pipeline([
    #     ('feature_selection', SelectFromModel(LinearSVC())),
    #     ('classification', DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0))
    # ])
    clf_dt = DecisionTreeClassifier(max_depth=None, min_samples_split=2,  random_state = 0)
    clf_dt.fit(X_train, y_train)

    # random forest
    # clf_rf = Pipeline([
    #     ('feature_selection', SelectFromModel(LinearSVC())),
    #     (
    #     'classification', RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0))
    # ])
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split = 2, random_state = 0)
    clf_rf.fit(X_train, y_train)


    lr = linear_model.LogisticRegression(C=1e5)
    lr.fit(X_train,y_train)




    # random forest classifier
    # neigh = Pipeline([
    #     ('feature_selection', SelectFromModel(LinearSVC())),
    #     ('classification',
    #      KNeighborsClassifier())
    # ])
    neigh = KNeighborsClassifier()
    neigh.fit(X_train, y_train)

    nb = GaussianNB()
    nb.fit(X_train,y_train)

    pt = Perceptron()
    pt.fit(X_train,y_train)

    # model_dir = model_dir+ '/traditional_models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # model_file = act_dir + category + '/' + fls[0] + '/model/' + fls[0]

    with open(model_dir + 'svm.model', 'wb') as fp:
        cPickle.dump(clf_svm, fp)

    with open(model_dir + 'linear_svm.model', 'wb') as fp:
        cPickle.dump(clf_linear_svm, fp)

    with open(model_dir + 'dt.model', 'wb') as fp:
        cPickle.dump(clf_dt, fp)

    with open(model_dir + 'rf.model', 'wb') as fp:
        cPickle.dump(clf_rf, fp)

    with open(model_dir + 'neigh.model', 'wb') as fp:
        cPickle.dump(neigh, fp)
    with open(model_dir + 'nb.model', 'wb') as fp:
        cPickle.dump(nb, fp)
    with open(model_dir + 'percepton.model', 'wb') as fp:
        cPickle.dump(pt, fp)
    with open(model_dir + 'LR.model', 'wb') as fp:
        cPickle.dump(lr, fp)


    test_classifier(X_test,y_test,filename=resultfile,model_dir=model_dir)

def write_results(fp, y, y_pred,methodname=''):
    cnf_mtx = confusion_matrix(y, y_pred)
    clas_wise_acc = cnf_mtx.diagonal() / cnf_mtx.sum(axis=0)
    acc = accuracy_score(y, y_pred)

    f1 = f1_score(y, y_pred, average=None)
    prc = precision_score(y, y_pred, average=None)
    rcl = recall_score(y, y_pred, average=None)
    print acc, f1, prc, rcl
    # fp.write(filename)
    res = [np.mean(acc),np.mean(f1),np.mean(prc),np.mean(rcl)]
    fp.write('Method= '+methodname)
    fp.write('\n\n')

    fp.write('\n\nAverage:\n')
    fp.write(str(res))
    fp.write('\n\n')

    fp.write("acc :: " + str(acc))
    fp.write('\n\n')
    fp.write('class wise:: '+','.join(str(x) for x in clas_wise_acc))
    fp.write('\n\n')
    fp.write('f1 :: ' + ','.join(str(x) for x in f1))
    fp.write('\n\n')
    fp.write("precision :: " + ','.join(
        str(x) for x in prc))
    fp.write('\n\n')
    fp.write("recall :: " + ','.join(str(x) for x in rcl))
    fp.write('\n\n\n')



def test_classifier(x=None,y=None,filename=None,model_dir=None):

    with open(model_dir + 'svm.model', 'rb') as fp:
        svm_model = cPickle.load( fp)

    with open(model_dir + 'linear_svm.model', 'rb') as fp:
        linear_svm_model = cPickle.load(fp)

    with open(model_dir + 'dt.model', 'rb') as fp:
        dt_model = cPickle.load(fp)

    with open(model_dir + 'rf.model', 'rb') as fp:
        rf_model = cPickle.load(fp)

    with open(model_dir + 'neigh.model', 'rb') as fp:
        neigh_model = cPickle.load(fp)

    with open(model_dir + 'nb.model', 'rb') as fp:
        nb_model = cPickle.load(fp)

    with open(model_dir + 'percepton.model', 'rb') as fp:
        pt_model = cPickle.load(fp)
    with open(model_dir + 'LR.model', 'rb') as fp:
        lr_model = cPickle.load(fp)

    fp.close()

    if filename:
        fp = open(filename,'w')

    y_pred = dt_model.predict(x)
    write_results(fp, y, y_pred, methodname='DT')

    y_pred = linear_svm_model.predict(x)
    write_results(fp, y, y_pred, methodname=' Linear SVM ')

    y_pred = svm_model.predict(x)
    write_results(fp, y, y_pred, methodname=' SVM ')

    y_pred = rf_model.predict(x)
    write_results(fp, y, y_pred, methodname=' Random Forest ')


    y_pred = neigh_model.predict(x)
    write_results(fp, y, y_pred, methodname=' KNN ')

    y_pred = nb_model.predict(x)
    write_results(fp, y, y_pred, methodname=' NB ')

    y_pred = pt_model.predict(x)
    write_results(fp, y, y_pred, methodname=' Percepton ')

    y_pred = lr_model.predict(x)
    write_results(fp, y, y_pred, methodname=' LR ')


    fp.close()

def build_classifier_model(trainfile, model_dir,filename):

    data = np.genfromtxt(trainfile, delimiter=',')
    row, col = data.shape
    train_features = data[:, 1:col - 1]
    train_labels = data[:, col - 1]
    # X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size = 0.2, random_state = 0)



    # svm train
    # clf_svm = Pipeline([
    #     ('feature_selection', SelectFromModel(LinearSVC())),
    #     ('classification', SVC())
    # ])
    # clf.fit(X, y)
    print 'building and training SVM ...'
    clf_svm = svm.SVC(decision_function_shape='ovo')
    clf_svm.fit(train_features, train_labels)

    print 'building and training Linear SVM ...'
    clf_linear_svm = svm.LinearSVC()
    clf_linear_svm.fit(train_features, train_labels)

    print 'building and training DT ...'
    clf_dt = DecisionTreeClassifier(max_depth=None, min_samples_split=2,  random_state = 0)
    clf_dt.fit(train_features, train_labels)

    print 'building and training RF ...'
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split = 2, random_state = 0)
    clf_rf.fit(train_features, train_labels)

    print 'building and training LR ...'
    lr = linear_model.LogisticRegression(C=1e5)
    lr.fit(train_features, train_labels)




    print 'building and training KNeighbors ...'
    neigh = KNeighborsClassifier()
    neigh.fit(train_features, train_labels)

    print 'building and training NB ...'
    nb = GaussianNB()
    nb.fit(train_features, train_labels)


    print 'building and training Perceptron ...'
    pt = Perceptron()
    pt.fit(train_features, train_labels)

    print 'training complete ...'

    print '\n\n'
    print 'saving models ...'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # model_file = act_dir + category + '/' + fls[0] + '/model/' + fls[0]
    name = filename.split('.')[0]
    model_dir=model_dir+name+'_'

    with open(model_dir + 'svm.model', 'wb') as fp:
        cPickle.dump(clf_svm, fp)

    with open(model_dir + 'linear_svm.model', 'wb') as fp:
        cPickle.dump(clf_linear_svm, fp)

    with open(model_dir + 'dt.model', 'wb') as fp:
        cPickle.dump(clf_dt, fp)

    with open(model_dir + 'rf.model', 'wb') as fp:
        cPickle.dump(clf_rf, fp)

    with open(model_dir + 'neigh.model', 'wb') as fp:
        cPickle.dump(neigh, fp)
    with open(model_dir + 'nb.model', 'wb') as fp:
        cPickle.dump(nb, fp)
    with open(model_dir + 'percepton.model', 'wb') as fp:
        cPickle.dump(pt, fp)
    with open(model_dir + 'LR.model', 'wb') as fp:
        cPickle.dump(lr, fp)

    print 'saving complete ...'


def train_classifiers():
    model_dir = 'models/'
    train_file = 'feature/'
    for per in range(1,11):
        print 'processing '+str(per)+'% file'
        dirname =train_file+str(per)+'/'
        files = get_all_files(dirname) # get all the data files - if you have multiple datafiles
        for fname in files:
            trainfile = os.path.join(dirname,fname) # get the training file for traditional classifier
            mdir = os.path.join(model_dir,str(per))
            print trainfile
            print model_dir

            build_classifier_model(trainfile, mdir,fname)




def test_classifiers():
    model_dir= 'models/'

    test_file = 'feature/clean_features.csv'
    result_dir = 'result/'
    result_file = 'result.csv'
    model_files = get_all_files(model_dir)

    for mfile  in model_files:
        model_file = os.path.join(model_dir,mfile) #get the path of the model file

        with open(model_file, 'rb') as fp:
            trained_model = cPickle.load(fp)  # load trained model
        data = np.genfromtxt(test_file, delimiter=',') # read the test file
        row, col = data.shape
        x = data[:, 1:col - 1]  # get data without the label information
        y_true = data[:, col - 1] # get the label information
        y_pred = trained_model.predict(x) # perform prediction

        acc = accuracy_score(y_true, y_pred)

        f1 = f1_score(y_true, y_pred, average=None)
        prc = precision_score(y_true, y_pred, average=None)
        rcl = recall_score(y_true, y_pred, average=None)


        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        res_file = os.path.join(result_dir,result_file)

        with open(res_file,'w') as fp:
            print 'writing individual results...'
            fp.write("acc :: " + str(acc))
            fp.write('\n\n')

            fp.write('f1 :: ' + ','.join(str(x) for x in f1))
            fp.write('\n\n')

            fp.write("precision :: " + ','.join(
                str(x) for x in prc))
            fp.write('\n\n')

            fp.write("recall :: " + ','.join(str(x) for x in rcl))
            fp.write('\n\n\n')



if __name__ == "__main__":

    train_classifiers()
    test_classifiers()
