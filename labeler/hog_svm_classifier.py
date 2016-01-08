import numpy
import skimage
from numpy import argmax
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold

# TODO: Consider http://scikit-learn.org/stable/modules/{sgd, kernel_approximation}.html for incremental SVM

class HogSvmClassifier(object):
    def __init__(self, labels, width, height, folds=5):
        self.labels = labels
        self.width = width
        self.height = height
        self.folds = folds
        self.classifiers = []
        self.scores=[]
        self.classifier = None
        """ The winning (best) classifier
        :type: LinearSVC
        """

        self.orientaions = 9
        """ Number of histogram bins for orientations
        :type: int
        """

        self.pixels_per_call = (8, 8)
        """ Number of pixels in each cell (each cell is replace by a histogram)
        :type: tuple[int, int]
        """

        self.cells_per_block = (3, 3)
        """  The number of cells to use when normalizing (to improve robustness to illumination)
        :type: tuple[int, int]
        """

        self.normalize = True
        """ Whether to normalize the input image (to improve robustness to illumination)
        :type: Boolean
        """

    def extract_features(self, sample):
        kwargs = dict(normalize=self.normalize,
                      orientations=self.orientations,
                      pixels_per_cell=self.pixels_per_cell,
                      cells_per_block=self.cells_per_block)

        features = hog(sample[:, :, 0], **kwargs)
        features += hog(sample[:, :, 1], **kwargs)
        features += hog(sample[:, :, 2], **kwargs)

        return features

    def train_offline(self, samples, labels):

        # labels is an array of int (0, 1, 2...)
        # samples is a generator for the raw images

        features = [self.extract_features(sample) for sample in samples]
        features = numpy.asarray(features)

        skf = StratifiedKFold(labels=labels, n_folds=self.folds, shuffle=True)

        scores = [0]*len(skf)
        classifiers = [None]*len(skf)

        for i, (train_index, test_index) in enumerate(skf):
            train_data = features[train_index]
            test_data = features[test_index]
            train_labels = labels[train_index]
            test_labels = labels[test_index]

            classifiers[i] = LinearSVC()
            classifiers[i].fit(train_data, train_labels)
            scores[i] = classifiers[i].score(test_data, test_labels)

        self.scores = scores
        self.classifiers = classifiers
        best = argmax(self.scores)
        self.classifier = self.classifiers[best]

    def classify(self, sample, which=None, vector=None):
        # I am keeping several classifiers, so that I can judge whether a sample is in a difficult area to classify
        if which is None and  self.classifier is None:
            return 0.5
        elif which is None:
            classifier = self.classifier
        else:
            classifier = self.classifiers[which]

        # I occasionally call this method several times, I do not want to repeat the feature extraction process
        if vector is None:
            features = self.extract_features(sample)
        else:
            features = vector

        label = classifier.predict([features])[0]
        return label

    def certainty(self, sample):
        features = self.extract_features(sample)
        result = 0

        label = self.classify(sample)
        for i in range(len(self.classifiers)):
            label = self.classifiers[i].predict([features])[0]


if __name__ == '__main__':

