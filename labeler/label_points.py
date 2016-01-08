import os
from math import sqrt, exp, ceil, floor
import gdal
import numpy
import pickle
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt4 import QtGui
from PyQt4.Qt import Qt
from PyQt4.QtGui import QApplication, QPen
from skimage.transform import rotate
from labeler.dataset import DataSet


class Classifier:
    def __init__(self, label, shape):
        self.label = label
        self.shape = shape
        self.positives = []
        self.negatives = []
        self.rbf = lambda x: exp(-x**2)
        self.transforms = []
        self.transforms.append(lambda x: rotate(x, 10))
        self.transforms.append(lambda x: rotate(x, 20))
        self.transforms.append(lambda x: rotate(x, 30))
        self.transforms.append(lambda x: rotate(x, 40))
        self.transforms.append(lambda x: rotate(x, 50))
        self.transforms.append(lambda x: rotate(x, 50))
        self.transforms.append(lambda x: rotate(x, 60))
        self.transforms.append(lambda x: rotate(x, 70))
        self.transforms.append(lambda x: rotate(x, 80))
        self.transforms.append(lambda x: x[::-1, :, :])
        self.transforms.append(lambda x: x[:, ::-1, :])
        self.transforms.append(lambda x: x + numpy.random.randn(*x.shape)*0.1)
        self.transforms.append(lambda x: x)
        self.filename = 'truth.pkl'

    def rebuild(self, dataset):
        for s in dataset.labeled('positive'):
            self.add_positive(s.data)
        for s in dataset.labeled('negative'):
            self.add_negative(s.data)

    def sample_width(self):
        return self.shape[0]

    def sample_height(self):
        return self.shape[1]

    def rbf(self, r):
        return 1./(abs(r)+1)

    def extract_features(self, sample):
        return sample.flatten()

    def compare(self, query, sample):
        diff = numpy.linalg.norm(query - sample)/len(query)
        weight = self.rbf(diff)
        return weight

    def add_negative(self, sample, transforms=None):
        if transforms is None: transforms = self.transforms
        for t in transforms:
            self.negatives.append(t(sample))

    def add_positive(self, sample, transforms=None):
        if transforms is None: transforms = self.transforms
        for t in transforms:
            self.positives.append(t(sample))

    def save(self, file=None):
        if file is None:
            file = open(self.filename, 'w')
        pickle.dump(dict(positives=self.positives, negatives=self.negatives), file)
        file.close()
        self.filename = file.name

    def update(self, file):
        data = pickle.load(file)
        self.positives += data['positives']
        self.negatives += data['negatives']
        self.filename = file.name

    def clear(self):
        self.positives = []
        self.negatives = []

    def load(self, file):
        self.clear()
        self.update(file)
        self.info()
        self.filename = file.name

    def find_similar_positive(self, sample):
        w_p = -1e16
        i_p = -1
        for i, p in enumerate(self.positives):
            w = self.compare(p, sample)
            if w > w_p:
                w_p = w
                i_p = i
        return i_p, w_p

    def find_similar_negative(self, sample):
        w_n = -1e16
        i_n = -1
        for i, n in enumerate(self.negatives):
            w = self.compare(n, sample)
            if w > w_n:
                w_n = w
                i_n = i
        return i_n, w_n

    def classify(self, sample):
        """
        :param sample:
        :return: > 0.5 = pos, < 0.5 = neg
        """
        i_p, w_p = self.find_similar_positive(sample)
        i_n, w_n = self.find_similar_negative(sample)
        return w_p / (w_p + w_n)

    def find_least_certain(self, samples):
        """Look through samples and return the index of the least certain, and its classification"""
        least_certain = 1.0
        least_certain_index = 0
        for i, s in enumerate(samples):
            w = self.classify(s)
            w += numpy.random.randn()*0.001
            if abs(w-0.5) < abs(least_certain-0.5):
                least_certain = w
                least_certain_index = i

        return least_certain_index, least_certain

    def info(self):
        print "positives:", len(self.positives)
        print "negatives:", len(self.negatives)





undos = []

MODE_POSITIVES = 'P'
MODE_NEGATIVES = 'N'
MODE_UNCERTAIN = 'U'

class LabelerWindow(QtGui.QWidget):
    def __init__(self, truth='truth', width=30, height=30, classifier = None):
        super(LabelerWindow, self).__init__()

        self.dataset = DataSet(truth, width, height)
        """ The ground truth images
        :type: labeler.dataset.DataSet
        """

        self.classifier = classifier or Classifier('cars', (width, height))
        """ The classifier we are building"""

        self.raster = None
        """The GDAL dataset we are viewing.
        :type ds: gdal.Dataset
        """

        self.geo_x = None
        """ The geographic x (longitude) of the center of the view """

        self.geo_y = None
        """ The geographic y (latitude) of the center of the view """

        self.scale = 1.0
        """ The number of meters per pixel in the view """

        self.buffer = None
        """ The raster data in the visible region (possibly padded with zeros)
        :type buffer: numpy.ndarray
        """

        self.random_walk = True
        """ Whether to look for the next sample within the current window (True) or within the entire raster (False)
        :type: Boolean
        """

        self.mode = MODE_POSITIVES

    def open_raster(self, datasource):
        """Load a raster dataset to display it in this window.

        This can be a single GeoTIFF (or other GDAL file) or a VRT file that mosaics several rasters.

        :param datasource: The GDAL datasource, or a description (filename)
        :type datasource: str | gdal.DataSet
        """

        if isinstance(datasource, gdal.Dataset):
            self.raster = datasource
        else:
            self.raster = gdal.Open(datasource)

    def visible_region(self, geo_x=None, geo_y=None, width=None, height=None, scale=None):
        if self.raster is None: return
        if geo_x is None: geo_x = self.geo_x
        if geo_y is None: geo_y = self.geo_y
        if width is None: width = self.width()
        if height is None: height = self.height()
        if scale is None: scale = self.scale

        assert geo_x is not None and geo_y is not None

        geo_transform = self.raster.GetGeoTransform()   # col, row --->  lon, lat
        ok, inverse_geo_transform = gdal.InvGeoTransform(geo_transform)  # lon, lat --->  col, row
        col, row = gdal.ApplyGeoTransform(inverse_geo_transform, self.geo_x, self.geo_y)

        # col, row are now where we are centered in the image
        # I want scale=1 to be viewing the image at 1m per pixel.
        # The image may be sampled at a different resolution, fetch the meters-per-pixel from the geo-transform
        # Also, the pixels generally are not square.
        meters_x = geo_transform[1]
        meters_y = geo_transform[5]

        # Determine the size and offset within the raster
        x_size = abs(int(width * meters_x / scale))
        y_size = abs(int(height * meters_y / scale))
        x_off = int(col - x_size/2)
        y_off = int(row - y_size/2)

        return x_off, y_off, x_size, y_size

    def load_buffer(self, geo_x=None, geo_y=None, width=None, height=None, scale=None):
        if self.raster is None: return
        x_off, y_off, x_size, y_size = self.visible_region(geo_x, geo_y, width, height, scale)

        # Load the data (gdal may keep it cached / memory mapped for us)
        clipped_x = max(0, -x_off)
        clipped_y = max(0, -y_off)
        clipped_xmax = min(self.raster.RasterXSize, x_off + x_size)
        clipped_ymax = min(self.raster.RasterYSize, y_off + y_size)
        clipped_width = clipped_xmax - clipped_x - x_off
        clipped_height = clipped_ymax - clipped_y - y_off

        if clipped_width < 0 or clipped_height < 0 or clipped_x >= x_size or clipped_y >= y_size:
            return numpy.zeros((y_size, x_size, self.raster.RasterCount), dtype=numpy.uint8)

        data = self.raster.ReadAsArray(x_off + clipped_x, y_off + clipped_y, clipped_width, clipped_height) \

        # Convert data into visible representation
        data = data.transpose(1, 2, 0)
        if not data.dtype == numpy.uint8:
            data += data.min()
            data *= 255/data.max()
            data = data.astype(numpy.uint8)

        if not data.shape[:2] == (y_size, x_size):
            padding = numpy.zeros((y_size, x_size, self.raster.RasterCount), dtype=numpy.uint8)
            padding[clipped_y:clipped_y+clipped_height, clipped_x:clipped_x+clipped_width, :] = data
            data = padding

        return data

    def center_of_raster(self):
        geo_transform = self.raster.GetGeoTransform()   # col, row --->  lon, lat
        x, y = gdal.ApplyGeoTransform(geo_transform, self.raster.RasterXSize / 2, self.raster.RasterYSize / 2)
        return x, y

    def look_at(self, geo_x, geo_y, scale=None):
        self.geo_x = geo_x
        self.geo_y = geo_y
        if scale is not None: self.scale = scale

        self.update_raster()

    def window_to_raster(self, x, y):
        rx, ry, rw, rh = self.visible_region()
        x = rx + rw*x/self.width()
        y = ry + rh*y/self.height()
        return x, y

    def raster_to_window(self, x, y):
        rx, ry, rw, rh = self.visible_region()
        x = -rx + x*self.width()/rw
        y = -ry + y*self.height()/rh
        return x, y

    def raster_to_geo(self, x, y):
        geo_transform = self.raster.GetGeoTransform()   # col, row --->  lon, lat
        x, y = gdal.ApplyGeoTransform(geo_transform, x, y)
        return x, y

    def geo_to_raster(self, x, y):
        geo_transform = self.raster.GetGeoTransform()   # col, row --->  lon, lat
        inv_transform = gdal.InvGeoTransform(geo_transform)
        x, y = gdal.ApplyGeoTransform(inv_transform, x, y)
        return x, y

    def geo_to_window(self, x, y):
        return self.raster_to_window(self.geo_to_raster(x, y))

    def window_to_geo(self, x, y):
        return self.raster_to_geo(*self.window_to_raster(x,y))

    def update_raster(self):
        self.buffer = self.load_buffer()
        self.update()

    def load_classifier(self):
        try:
            if os.path.exists('training.pkl'):
                with open('training.pkl', 'rb') as f:
                    self.classifier.load(f)
        except:
            self.classifier.rebuild(self.dataset)

    def paintEvent(self, QPaintEvent):
        super(LabelerWindow, self).paintEvent(QPaintEvent)

        p = QtGui.QPainter(self)
        image = ImageQt(Image.fromarray(self.buffer))
        image = image.scaled(self.width(), self.height())
        p.drawImage(0, 0, image)

        # Now for the HUD

        # -> Draw green cross-hairs
        old_pen = p.pen()
        new_pen = QPen()
        new_pen.setColor(Qt.green)
        new_pen.setStyle(Qt.DotLine)
        new_pen.setWidth(1)
        p.setPen(new_pen)
        p.drawLine(0, self.height()/2, self.width(), self.height()/2)
        p.drawLine(self.width()/2, 0, self.width()/2, self.height())
        p.setPen(old_pen)

        # -> Show help for keystrokes
        help = "[X] Pos. [C] Neg. [UP] Zoom in [DN] Zoom Out [LT] Undo Last [RT] Ignore [LMB] Move "
        p.fillRect(0, 0, self.width(),  p.fontMetrics().height()*1.5, Qt.gray)
        p.drawText(0, p.fontMetrics().height(),  help)

    def resizeEvent(self, QResizeEvent):
        self.update_raster()

    def get_sample(self):
        s = self.load_buffer(width=self.dataset.width, height=self.dataset.height, scale=0.25).astype(float)/255.0
        return s

    def geo_xy(self):
        return self.geo_x, self.geo_y

    def goto_next_sample(self):
        if self.mode == MODE_POSITIVES:
            self.goto_next_positive()
        elif self.mode == MODE_NEGATIVES:
            self.goto_next_negative()
        else:
            self.goto_next_uncertain()

    def get_random_xys(self, n):
        if self.random_walk:
            return self.get_random_in_view(n)
        else:
            return self.get_random_in_raster(n)

    def get_random_in_raster(self, n):
        width, height = self.classifier.shape
        x = numpy.random.random_integers(int(ceil(width/2)), int(floor(self.raster.RasterXSize - width / 2)), n)
        y = numpy.random.random_integers(int(ceil(height/2)), int(floor(self.raster.RasterYSize - height / 2)), n)

        return zip(x, y)

    def get_random_in_view(self, n):
        x0, y0, w, h = self.visible_region()
        x1, y1 = x0+w, y0+h
        width, height = self.classifier.shape
        rx, ry = width / 2, height/2
        x0 = int(ceil(max(x0, rx)))
        y0 = int(ceil(max(y0, ry)))
        x1 = int(floor(min(self.raster.RasterXSize-rx, x1)))
        y1 = int(floor(min(self.raster.RasterYSize-ry, y1)))
        x = numpy.random.random_integers(x0, x1, n)
        y = numpy.random.random_integers(y0, y1, n)
        return zip(x, y)



    def goto_next_uncertain(self):
        x, y = zip(*self.get_random_xys(100))
        samples = []
        for i in range(len(x)):
            self.look_at(*self.raster_to_geo(x[i], y[i]))
            s = self.get_sample()
            samples.append(s)
        i, w = self.classifier.find_least_certain(samples)

        print "Least confident:", w
        self.look_at(*self.raster_to_geo(x[i], y[i]))

    def goto_next_positive(self, thresh=-1):
        best = -1
        best_x, best_y = self.geo_x, self.geo_y

        for i, (x, y) in enumerate(self.get_random_xys(100)):
            self.look_at(*self.raster_to_geo(x, y))
            s = self.get_sample()
            w = self.classifier.classify(s)
            if w > thresh:
                best = i
                best_x, best_y = x, y
                thresh = w
        self.look_at(*self.raster_to_geo(best_x, best_y))


    def goto_next_negative(self, thresh=0.5):
        for i, (x, y) in enumerate(self.get_random_xys(100)):
            self.look_at(*self.raster_to_geo(x, y))
            s = self.get_sample()
            w = self.classifier.classify(s)
            if w < thresh:
                print "Negative sample", w
                break

    def keyPressEvent(self, QKeyEvent):
        key = QKeyEvent.key()

        x, y = self.geo_x, self.geo_y

        def undo_positive():
            self.look_at(x, y)
            self.dataset.delete(self.get_sample(), label='positive')
            self.classifier.positives.pop()
            self.classifier.save()

        def undo_negative():
            self.look_at(x, y)
            self.dataset.delete(self.get_sample(), label='negative')
            self.classifier.negatives.pop()
            self.classifier.save()

        if key == Qt.Key_X:
            print "Mark Pos. ", self.raster.GetDescription(), self.geo_xy()
            self.classifier.add_positive(self.get_sample())
            self.dataset.create(self.get_sample(), label='positive', meta=dict(pos=self.geo_xy())).save()
            self.classifier.save()
            undos.append(undo_positive)
            self.goto_next_sample()
        elif key == Qt.Key_C:
            print "Mark Neg. ", self.raster.GetDescription(), self.geo_xy()
            self.classifier.add_negative(self.get_sample())
            self.dataset.create(self.get_sample(), label='negative', meta=dict(pos=self.geo_xy())).save()
            self.classifier.save()
            undos.append(undo_negative)
            self.goto_next_sample()
        elif key == Qt.Key_Left:
            print len(undos)
            if len(undos) > 0:
                undos[-1]()
                undos.pop()
        elif key == Qt.Key_Right:
            print "Next Sample (do not save current)"
            self.goto_next_sample()
        elif key == Qt.Key_Up:
            self.scale *= sqrt(2.)
            self.update_raster()
        elif key == Qt.Key_Down:
            self.scale /= sqrt(2.)
            self.update_raster()
        elif key == Qt.Key_S:
            self.save_classifier()
        elif key == Qt.Key_R:
            self.classifier.rebuild(self.dataset)
            self.save_classifier()
            self.goto_next_sample()
        elif key == Qt.Key_L:
            self.load_classifier()
        elif key == Qt.Key_P:
            self.mode = MODE_POSITIVES
            self.goto_next_sample()
        elif key == Qt.Key_N:
            self.mode = MODE_NEGATIVES
            self.goto_next_sample()
        elif key == Qt.Key_U:
            self.mode = MODE_UNCERTAIN
            self.goto_next_sample()
        elif key == Qt.Key_V:
            s = self.get_sample()
            self.visualize_classification(s)

    def save_classifier(self):
        self.classifier.save(file('training.pkl', 'wb'))

    def visualize_classification(self, s):
        import pylab
        i_p, w_p = self.classifier.find_similar_positive(s)
        i_n, w_n = self.classifier.find_similar_negative(s)
        pylab.figure()
        pylab.subplot(231)
        if i_n >= 0:
            pylab.imshow(self.classifier.negatives[i_n])
        pylab.title('negative')
        pylab.subplot(232)
        pylab.imshow(s)
        pylab.title('query')
        pylab.subplot(233)
        if i_p >= 0:
            pylab.imshow(self.classifier.positives[i_p])
        pylab.title('positive')
        pylab.subplot(234)
        if i_n >= 0:
            pylab.imshow((abs(self.classifier.negatives[i_n].astype(float) - s) ** 2).mean(2), cmap=pylab.cm.gray,
                         vmax=255 ** 2)
        pylab.subplot(236)
        if i_p >= 0:
            pylab.imshow((abs(self.classifier.positives[i_p].astype(float) - s) ** 2).mean(2), cmap=pylab.cm.gray,
                         vmax=255 ** 2)
        pylab.show()

    def mousePressEvent(self, QMouseEvent):
        x, y = QMouseEvent.x(), QMouseEvent.y()
        button = QMouseEvent.button()

        if button == Qt.LeftButton:
            rx, ry = self.window_to_raster(x, y)
            gx, gy = self.raster_to_geo(rx, ry)
            self.look_at(gx, gy)
            QtGui.QCursor.setPos(self.mapToGlobal(self.rect().center()))




if __name__ == '__main__':
    app = QApplication([])
    win = LabelerWindow()

    #URLS:   https://drive.google.com/open?id=0B8AGWB7NYzjXQjhkVm9iWFlUaU0
    #URLS:   https://drive.google.com/open?id=0B8AGWB7NYzjXQk5NTS1ibFJYM3c

    win.open_raster(r'D:\sf_building_footprints\201104_san_francisco_ca_0x3000m_utm_clr\north_up\10seg460805.tif')
    win.load_classifier()
    win.look_at(*win.center_of_raster())
    win.goto_next_uncertain()
    win.show()
    app.exec_()


