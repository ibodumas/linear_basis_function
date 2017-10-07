#################### This cell is for all the methods and data import ################################
import os
import sklearn.model_selection
import pandas
from numpy.linalg import inv  # matrix inverse
import random  # for seeding and random no generation
from itertools import chain  # for unlisting
import matplotlib.pyplot as matplot
from numpy import linalg
import numpy

# stratification
def folds_stratify(nSample, seed, kFolds):  # this return kfold stratification
    random.seed(seed)
    foldSize = round(nSample / kFolds, 0)
    randomList = list(range(0, nSample))
    random.shuffle(randomList)
    stra = []

    for k in range(0, kFolds):
        strt = int(k * foldSize)
        end = int((k + 1) * foldSize)
        if k == (kFolds - 1):
            end = nSample
        stra.append(list(randomList)[strt:end])

    return stra
# end def folds_stratify

# linear basis function
def W_lbf(trainX, noutputs, lamda):  # linear basis function# lamda = reqularization coefficient
    nFeature = trainX.shape[1] - noutputs
    trainX_p = trainX[:, 0:nFeature]
    trainX_p = pandas.DataFrame(trainX_p)
    X = pandas.concat([trainX_p[0], trainX_p], axis=1)  # Adding one column to X
    X.iloc[:, 0] = 1  # setting x0 = 1, from the column added above
    X = numpy.asarray(X)
    phi = X  # in linear basis function #ϕ
    y = trainX[:, nFeature:nFeature + noutputs]  # the last noutputs columns in trainX
    phi_trans = phi.transpose()  # phi transpose
    phi_trans_phi = numpy.dot(phi_trans, phi)
    I = numpy.identity(phi_trans_phi.shape[0])  # add 1 to nFeature bcos of x0
    lamda_I = lamda * I
    add_lamda_I_phi_trans_phi = numpy.add(lamda_I, phi_trans_phi)
    inv_sum = inv(add_lamda_I_phi_trans_phi)  # inverse the matrix above
    inv_sum_phi_trans = numpy.dot(inv_sum, phi_trans)
    W = numpy.dot(inv_sum_phi_trans, y)
    return W
# end def W_lbf

# cross validation for linear basis function
def cv_lbf(data, noutputs, kFolds, lamda, seed):
    nFeature = data.shape[1] - noutputs
    nSample = data.shape[0]
    stra_all = folds_stratify(nSample=nSample, seed=seed, kFolds=kFolds)
    df = pandas.DataFrame(index=list(range(0, len(lamda))), columns=list(range(0, noutputs + 1)))

    for index, val in enumerate(lamda):
        df.iloc[index, 0] = val
        error_per_y = pandas.DataFrame(index=list(range(0, kFolds)), columns=list(range(0, noutputs)))
        k = 0
        while k < kFolds:
            stra = stra_all.copy()
            test = data[stra[k]]
            del stra[k]  # del test list
            stra_train = list(chain.from_iterable(stra))  # merge the sublists
            train = data[stra_train]
            w_vals = W_lbf(trainX=train, noutputs=noutputs, lamda=val)
            # w_kFolds.append(w_vals)
            x_test = test[:, 0:nFeature]
            x_test = pandas.DataFrame(x_test)
            x_test = pandas.concat([x_test[0], x_test], axis=1)  # Adding one column to X
            x_test.iloc[:, 0] = 1  # setting x0 = 1, from the column added above
            x_test = numpy.asarray(x_test)
            y_actual = test[:, nFeature:(nFeature + noutputs)]
            y_pred = numpy.dot(x_test, w_vals)
            if y_actual.shape != y_pred.shape:
                print("\n\nError002: Shape not equal: y_actual.shape != y_pred.shape\n\n")
            # y_actual_pred = y_actual - y_pred
            y_actual_pred = numpy.subtract(y_actual, y_pred)
            error_2 = numpy.square(y_actual_pred)
            errors = numpy.sum(error_2, axis=0)
            i = 0
            # compute error for each y
            while i < noutputs:
                error_per_y.iloc[k, i] = errors[i]
                i += 1
            k += 1
        j = 0
        while j < noutputs:
            df.iloc[index, j + 1] = numpy.average(error_per_y.iloc[:, j], axis=0)
            j += 1
    return df
# end def W_lbf

# cross validation for linear basis function
def cv_lbf_all_data(data, noutputs, kFolds, lamda, seed):
    nFeature = data.shape[1] - noutputs
    nSample = data.shape[0]
    stra_all = folds_stratify(nSample=nSample, seed=seed, kFolds=kFolds)
    df = pandas.DataFrame(index=["error"], columns=list(range(0, noutputs)))

    index = 0
    errors_per_y = []
    while index < noutputs:
        error_per_fold = []
        val = lamda[index]
        k = 0
        while k < kFolds:
            stra = stra_all.copy()
            test = data[stra[k]]
            del stra[k]  # del test list
            stra_train = list(chain.from_iterable(stra))  # merge the sublists
            train = data[stra_train]
            train = pandas.DataFrame(train)
            x = train.iloc[:, 0:nFeature]
            y = train.iloc[:, nFeature + index]  # get the current y
            trainX_new = numpy.asarray(x.join(y))
            y_actual = test[:, nFeature + index]  # get the current y
            x_test = test[:, 0:nFeature]
            x_test = pandas.DataFrame(x_test)
            x_test = pandas.concat([x_test[0], x_test], axis=1)  # Adding one column to X
            x_test.iloc[:, 0] = 1  # setting x0 = 1, from the column added above
            x_test = numpy.asarray(x_test)
            w_vals = W_lbf(trainX=trainX_new, noutputs=1, lamda=val)
            y_pred = numpy.dot(x_test, w_vals)
            y_actual.shape = (y_actual.shape[0], 1)
            if y_actual.shape != y_pred.shape:
                print("\n\nError002: Shape not equal: y_actual.shape != y_pred.shape\n\n")
            # y_actual_pred = y_actual - y_pred
            y_actual_pred = numpy.subtract(y_actual, y_pred)
            error_2 = numpy.square(y_actual_pred)
            errors = numpy.sum(error_2)
            error_per_fold.append(errors)
            k += 1
        index += 1
        errors_per_y.append(numpy.mean(error_per_fold))
    return [errors_per_y, numpy.sum(errors_per_y)]
# end def cv_lbf

# main lbf function
def lbf_main(trainX, testX, noutputs, nFeature):
    kFolds = 5
    trainX_pandas = pandas.DataFrame(trainX)
    lamda = [0, 0.001, 0.01, 0.1, 10, 100, 1000]  # λ
    seed = 3221226
    df_lbf = cv_lbf(data=trainX, noutputs=noutputs, kFolds=kFolds, lamda=lamda, seed=seed)  # do CV to pick the best lamda
    print("\nTable of average error(kFold CV) per lamda per the target variable(s)\n" + df_lbf.to_string() + "\n")
    lamda = df_lbf.iloc[:, 0]
    column = ['best_lamda_per_y', 'error']
    df_best_lbf = pandas.DataFrame(index=list(range(0, noutputs)), columns=column)

    u = 0
    while u < noutputs:
        err_al = list(df_lbf.iloc[:, u + 1])
        minerr = min(err_al)
        ind_best_lam = err_al.index(minerr)
        df_best_lbf.iloc[u, 0] = lamda[ind_best_lam]
        df_best_lbf.iloc[u, 1] = minerr
        u += 1
        # end while

    print("\nTable of best lamda per the target variable(s) - {the index correspond to the y(s)}\n" + df_best_lbf.to_string() + "\n")
    columns_lbf = ['y', 'error_per_y', 'W_per_y']
    df_lbf_final = pandas.DataFrame(index=list(range(0, noutputs)), columns=columns_lbf)
    # df_lbf_final['y'] = list(range(noutputs))

    print("\nHaving chosen the best set(s) of lamda:")
    print("Below are the analysis of training the best parameters on trainX and evaluating on testX:\n")

    joins = []
    u_lbf = 0
    while u_lbf < noutputs:
        x = trainX_pandas.iloc[:, 0:nFeature]
        y = trainX_pandas.iloc[:, nFeature + u_lbf]  # get the current y
        trainX_new = numpy.asarray(x.join(y))
        w_vals_lbf = W_lbf(trainX=trainX_new, noutputs=1, lamda=df_best_lbf["best_lamda_per_y"][u_lbf])
        x_test_lbf = testX[:, 0:nFeature]
        x_test_lbf = pandas.DataFrame(x_test_lbf)
        x_test_lbf = pandas.concat([x_test_lbf[0], x_test_lbf], axis=1)  # Adding one column to X
        x_test_lbf.iloc[:, 0] = 1  # setting x0 = 1, from the column added above
        x_test_lbf = numpy.asarray(x_test_lbf)
        y_actual_lbf = testX[:, nFeature + u_lbf]
        y_actual_lbf.shape = (y_actual_lbf.shape[0], 1)
        y_pred_lbf = numpy.dot(x_test_lbf, w_vals_lbf)
        y_actual_pred = numpy.subtract(y_actual_lbf, y_pred_lbf)
        error_2_lbf = numpy.square(y_actual_pred)
        error = sum(error_2_lbf)  # / 2
        df_lbf_final['error_per_y'][u_lbf] = error
        df_lbf_final['W_per_y'][u_lbf] = w_vals_lbf
        df_lbf_final['y'][u_lbf] = u_lbf

        join1 = pandas.DataFrame(y_actual_lbf, columns=['y_actual_lbf'])
        join2 = pandas.DataFrame(y_pred_lbf, columns=['y_pred_lbf'])
        join3 = pandas.DataFrame(y_actual_pred, columns=['y_actual_pred'])
        join4 = pandas.DataFrame(error_2_lbf, columns=['error_2_lbf'])
        join = pandas.concat([join1, join2, join3, join4], axis=1)
        joins.append(join)
        print("\n" + "Summary table of test data relating to y{}\n".format(u_lbf) + join.head(5).to_string() + "\n")
        # plot
        print("y_actual vs. predict for variable y{} \n".format(u_lbf))
        matplot.scatter(y_actual_lbf, y_pred_lbf)
        matplot.xlabel('y_actual_lbf')
        matplot.ylabel('y_pred_lbf')
        matplot.show()
        u_lbf += 1
        # end while

    print(df_lbf_final.to_string() + "\n")
    print("The total error = {}".format(sum(chain.from_iterable(df_lbf_final["error_per_y"]))))
    return [df_lbf_final, joins, df_best_lbf["best_lamda_per_y"]]
# end lbf_main

# my_regression
def my_regression(trainX, testX, noutputs):
    columns = None
    row = ["best_params", "best_error"]
    if noutputs == 1:
        columns = ["y0"]
    elif noutputs == 3:
        columns = ["y0", "y1", "y2"]

    nFeature = trainX.shape[1] - noutputs  # No of features
    df_lbf = lbf_main(trainX, testX, noutputs, nFeature)

    return [df_lbf]
# end def my_regression

#CV for all dataset
def CV_on_all_data_lbf(allData, lbf_param, noutputs, dataName):
    kFolds = 5
    lamda_lbf = lbf_param
    seed_lbf = 650932
    cv_all_data_lbf = cv_lbf_all_data(data=allData, noutputs=noutputs, kFolds=kFolds, lamda=numpy.asarray(lbf_param[2]), seed=seed_lbf)
    print("Linear BF ERROR for " + dataName + " = {}".format(cv_all_data_lbf[1]))
    return [cv_all_data_lbf]

####################################### Import Data #########################################
os.chdir('C:/Users/2PAC/Documents/Python Scripts/pycharm/ML/HW1_Regression') #set new directory
def z_score_norm(data):
    if type(data) is numpy.ndarray:
        mean = numpy.mean(data, axis=0)
            #data.mean()
        std = numpy.std(data, axis=0) #data.std()
        data_norm = (data - mean) / std
        result = data_norm
    else:
        result = "Error001: Provide numpy array"

    return result

'''AIRFOIL'''
#from numpy import loadtxt
airfoil = numpy.loadtxt("airfoil_self_noise.dat.txt")
sample_size_af = airfoil.shape[0]
airfoil_norm = z_score_norm(airfoil)
random.seed(5054123) #set seed
x_train_af, x_test_af = sklearn.model_selection.train_test_split(airfoil_norm, test_size=0.2, random_state=0)
noutputs_af = 1

'''YACHT'''
yacht = numpy.loadtxt("yacht_hydrodynamics.data.txt")
sample_size_yt = yacht.shape[0]
yacht_norm = z_score_norm(yacht)
random.seed(3452332)
x_train_yt, x_test_yt = sklearn.model_selection.train_test_split(yacht_norm, test_size=0.2, random_state=0)
noutputs_yt = 1

'''SLUMP'''
slump = numpy.loadtxt("slump_test.data.txt", skiprows=1, delimiter=",")
slump = slump[:,1:11]
sample_size_sp = slump.shape[0]
slump_norm = z_score_norm(slump)
random.seed(3450423)
x_train_sp, x_test_sp = sklearn.model_selection.train_test_split(slump_norm, test_size=0.2, random_state=0)
noutputs_sp = 3
####################################### End Import Data #########################################

#################### Airfoil Data - Linear Basis Function ################################
myReg_airfoil = my_regression(trainX=x_train_af, testX=x_test_af, noutputs=noutputs_af)
print("\nCROSS VALIDATION OUTSIDE my_Regrssion FUNCTION\n5 FOLDS CROSS VALIDATION FOR ALL AIRFOIL DATA: RESULT")
cv_all_data_airfoil = CV_on_all_data_lbf(allData=airfoil_norm, lbf_param=myReg_airfoil[0], noutputs=noutputs_af, dataName="Airfoil")
#myReg__airfoil_jupyter = my_regression(trainX=x_train_af, testX=x_test_af, noutputs=noutputs_af)
#################### End of Airfoil Data  ################################

#################### Yacht Data - Linear Basis Function ################################
myReg_yacht = my_regression(trainX=x_train_yt, testX=x_test_yt, noutputs=noutputs_yt)
print("\nCROSS VALIDATION OUTSIDE my_Regrssion FUNCTION\n5 FOLDS CROSS VALIDATION FOR ALL YACHT DATA: RESULT")
cv_all_data_yacht = CV_on_all_data_lbf(allData=yacht_norm, lbf_param=myReg_yacht[0], noutputs=noutputs_yt, dataName="Yacht")
#################### End of Yacht Data  ################################


#################### Slump Data - Linear Basis Function ################################
myReg_slump = my_regression(trainX=x_train_sp, testX=x_test_sp, noutputs=noutputs_sp)
print("\nCROSS VALIDATION OUTSIDE my_Regrssion FUNCTION\n5 FOLDS CROSS VALIDATION FOR ALL SLUMP DATA: RESULT")
cv_all_data_slump = CV_on_all_data_lbf(allData=slump_norm, lbf_param=myReg_slump[0], noutputs=noutputs_sp, dataName="Slump")
#################### End of Slump Data  ################################
