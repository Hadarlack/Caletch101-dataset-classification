import os
import cv2
import numpy as np
import random
import skimage as ski
from skimage.feature import hog
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# there is random element in the code, therefore each run will produce different err rate
def GetDefaultParameters():
    params = {}
    params['data_path'] = "C:/Users/wosne/PycharmProjects/101_ObjectCategories"
    # indices start from 0 -> i.e for classes 1,2,3 the indices are 0,1,2!
    params['class_indices'] = list(range(10, 20))
    params['class_indices'] = [27,50,61,72,77,10,12,43,54,11,22,33,44,55,66]
    params['s'] = 105
    params['images_per_class'] = 40
    params['split'] = 20  # 20 examples for train
    params['num_bins'] = 10  # bins for HOG
    params['pixels_per_cell'] = 23
    params['cells_per_block'] = 2
    params['max_iter'] = 1000  # iteration fot the svm
    params['c'] = 10
    params['block_norm'] = 'L1'
    params['gamma'] = 1.7  # for the RBF kernel
    return params


# loading the data from the path
def get_data(data_path, class_indices, s, images_per_class):
    label_names = os.listdir(data_path)
    data = []
    labels = []
    for i in class_indices:
        path_dir = data_path + "/" + label_names[i]
        images = os.listdir(path_dir)
        images = random.sample(images, min(images_per_class, len(images)))  # shuffle the pictures order and selecting only 40
        for image in images:
            tmp = cv2.imread(path_dir + '/' + image)
            tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
            tmp = cv2.resize(tmp, (s, s))
            data.append(tmp)
            labels.append(i)
    data = np.asarray(data)
    labels = np.asarray(labels)
    dandl = {'data': data, 'labels': labels, 'label_names': label_names}
    return dandl


def TrainTestSplit(data, labels, class_indices, split, s):
    train_data = np.array([])
    train_label = np.array([])
    test_data = np.array([])
    test_labels = np.array([])
    inserted = 0
    for i in class_indices:
        train_data = np.append([train_data], [data[inserted:inserted + split]])
        train_label = np.append([train_label], [labels[inserted:inserted + split]])
        num_of_test = np.count_nonzero(labels == i) - split
        test_data = np.append([test_data], [data[inserted + split: inserted + split + num_of_test]])
        test_labels = np.append([test_labels], [labels[inserted + split: inserted + split + num_of_test]])
        inserted = inserted + split + num_of_test
    # reshaping the flatted arrays
    n_train = split * len(class_indices)
    train_data = np.reshape(train_data, (n_train, s, s))
    n_test = len(labels)-n_train
    test_data = np.reshape(test_data, (n_test, s, s))
    return train_data, train_label, test_data, test_labels


def prepare(data, num_bins, pixels_per_cell, cells_per_block, block_norm):
    data_featured = []
    for image in data:
        image_feature = ski.feature.hog(image, orientations=num_bins, pixels_per_cell=(pixels_per_cell, pixels_per_cell)
                                        , cells_per_block=(cells_per_block, cells_per_block),
                                        block_norm=block_norm, visualize=False, visualise=None,
                                        transform_sqrt=False, feature_vector=True, multichannel=None)
        data_featured.append(image_feature)
    data_featured = np.asarray(data_featured)
    return data_featured


def train(X, y, c, max_iter):
    clf = LinearSVC(C=c, multi_class='ovr', verbose=0, random_state=None, max_iter=max_iter)
    clf.fit(X, y)
    return clf


# return the error rate for the model and the data
def test(model, data, labels, class_indices):
    # for kernel SVM
    res, class_score_matrix = n_class_SVM_predict(data, model, class_indices)
    # for linear SVM
    # res = model.predict(data)

    err_rate = sum(res != labels)/labels.shape
    return err_rate, res , class_score_matrix


def n_class_SVM_train(class_indices, X, y, gamma, c):
    models = []
    # loop over each class and train according to this class
    for i in class_indices:
        # creating labels vector for the i class
        y_i = np.copy(y)
        y_i[y_i != i] = -1
        y_i[y_i == i] = 1
        # creating model
        clf = SVC(C=c, decision_function_shape='ovo',
                  gamma=gamma, kernel='rbf', max_iter=-1, probability=False,
                  shrinking=True, tol=0.001, verbose=False)
        # fit the model to data for class i
        clf.fit(X, y_i)
        models.append(clf)
    models = np.asarray(models)
    return models


def n_class_SVM_predict(X, models, class_indices):
    class_score_matrix = np.array([])
    for i in range(models.shape[0]):
        res = models[i].decision_function(X)
        class_score_matrix = np.append(class_score_matrix, res)
    class_score_matrix = np.reshape(class_score_matrix, (X.shape[0], models.shape[0]), order='F')
    prediction = class_score_matrix.argmax(axis=1)
    # converting the indexing from 0-9 to the correct labels
    for i in range(len(class_indices)):
        prediction[prediction == i] = class_indices[i]
    return prediction, class_score_matrix


def TrainWithTuning():
    block_norm = np.array(['L1', 'L1-sqrt', 'L2', 'L2-Hys'])
    block_norm = np.arange(150, 200, 10)

    err_rate = np.ones_like(block_norm, dtype=float)
    for i in range(block_norm.shape[0]):
        params['block_norm'] = block_norm[i]
        DandL = get_data(params['data_path'], params['class_indices'], params['s'], params['images_per_class'])
        TrainData, TrainLabels, TestData, TestLabels = TrainTestSplit(DandL['data'], DandL['labels'],
                                                                      params['class_indices'], params['split'],
                                                                      params['s'])
        # for linear and non linear tuning
        Train_featured = prepare(TrainData, params['num_bins'], params['pixels_per_cell'], params['cells_per_block'],
                                 params['block_norm'])
        Test_featured = prepare(TestData, params['num_bins'], params['pixels_per_cell'], params['cells_per_block'],
                                params['block_norm'])
        # non linear tuning
        model = n_class_SVM_train(params['class_indices'], Train_featured, TrainLabels, params['gamma'], params['c'])
        err_rate[i], res = test(model, Test_featured, TestLabels, params['class_indices'])
        # for linear tuning
        # model = train(Train_featured, TrainLabels, params['c'], params['max_iter'])
        # err_rate[i] = test(model, Test_featured, TestLabels)

    print(block_norm)
    print(err_rate)

    plt.scatter(block_norm, err_rate)
    plt.title("err_rate vs block_norm method")
    plt.xlabel("block_norm")
    plt.ylabel("err_rate")
    plt.show()
    return


def confused_pic(class_indices, y, prediction, decision_matrix):
    wrong_classify = y != prediction
    index_wrong = np.argwhere(wrong_classify == True)  # indices for misclassified example
    index_wrong= index_wrong[:, 0]
    y_wrong = y[wrong_classify]  # belonged class for the misclassified
    prediction_wrong = prediction[wrong_classify]  # the predicted class for the misclassified examples
    wrong_decision_matrix = decision_matrix[wrong_classify, :]  # only the rows that misclassified

    # mapping class to index -> the first class will map to 0 the second to 1 and go on...
    j = 0  # counter
    y_wrong_mapping = np.copy(y_wrong)
    prediction_wrong_mapping = np.copy(prediction_wrong)
    for i in class_indices:
        y_wrong_mapping[y_wrong_mapping == i] = j
        prediction_wrong_mapping[prediction_wrong_mapping == i] = j
        j = j + 1
    y_wrong_mapping = y_wrong_mapping.astype(int)
    prediction_wrong_mapping = prediction_wrong_mapping.astype(int)

    # create vector with the mistake value to the correct class from the predicted class
    mistakes_values = np.asarray([])
    for i in range(y_wrong.shape[0]):  # loop over each row of wrong_decision_matrix, all the misclassified
        mistake = wrong_decision_matrix[i, y_wrong_mapping[i]] - wrong_decision_matrix[i, prediction_wrong_mapping[i]]
        mistakes_values = np.append(mistakes_values, mistake)

    # combine mistake, belonged class, original index to one matrix
    combine = np.asarray([mistakes_values, y_wrong, index_wrong])
    combine = np.transpose(combine)
    # sorting the combined matrix by the mistake
    combine = combine[combine[:, 0].argsort()[::-1]]
    # sorting the combined matrix by the mistake and then by class
    combine = combine[combine[:, 1].argsort()]

    # find 2 largest mistakes
    example_idx = np.asarray([])
    belonged_class = np.asarray([])
    for i in range(combine.shape[0]):
        if example_idx.shape[0] >= 2:
            if belonged_class[-1] == combine[i, 1] and belonged_class[-2] == combine[i, 1]:
                continue
            else:
                example_idx = np.append(example_idx, combine[i, 2])
                belonged_class = np.append(belonged_class, combine[i, 1])
        else:
            example_idx = np.append(example_idx, combine[i, 2])
            belonged_class = np.append(belonged_class, combine[i, 1])

    # show the images
    for i in range(example_idx.shape[0]):
        plt.imshow(TrainData[example_idx.astype(int)][i], cmap = 'gray')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
    return


params = GetDefaultParameters()
DandL = get_data(params['data_path'], params['class_indices'], params['s'], params['images_per_class'])
TrainData, TrainLabels, TestData, TestLabels = TrainTestSplit(DandL['data'], DandL['labels'],
                                                              params['class_indices'], params['split'], params['s'])
# representation of the data
Train_featured = prepare(TrainData, params['num_bins'], params['pixels_per_cell'], params['cells_per_block'],
                         params['block_norm'])
Test_featured = prepare(TestData, params['num_bins'], params['pixels_per_cell'], params['cells_per_block'],
                        params['block_norm'])

# train
classifiers = n_class_SVM_train(params['class_indices'], Train_featured, TrainLabels, params['gamma'], params['c'])
# classifiers = train(Train_featured, TrainLabels, params['c'],params['max_iter'])

# prediction and summary
err_rate, prediction, class_score_matrix = test(classifiers, Test_featured, TestLabels, params['class_indices'])
confusion_matrix = confusion_matrix(TestLabels, prediction)

# result printing
print('Confusion matrix')
print(confusion_matrix)
print('%s%f' % ('The err rate for the non linear SVM is: ', err_rate))
# show largest err
confused_pic(params['class_indices'], TestLabels, prediction, class_score_matrix)


