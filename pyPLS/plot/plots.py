import numpy as np
from bokeh.plotting import figure
from bokeh.plotting import show
from .bokeh_pypls import bokeh_pypls
from .mplt_pypls import mplt_pypls
from pyPLS.utilities import ROCanalysis


class scatterPlot(object):
    def __init__(self, x, y, **kwargs):
        self.x = x
        self.y = y
        self.arguments = kwargs
        # TODO: make possible multicolumn Y to be plotted
        if x.ndim > 1:
            n, p = x.shape
            if p < n:
                if p > 1:
                    raise Exception('Data must be 1-dimensional')
                else:
                    x = x[:, 0]
            else:
                if n > 1:
                    raise Exception('Data must be 1-dimensional')
                else:
                    x = x[0, :]

        if y.ndim > 1:
            n, p = y.shape
            if p < n:
                if p > 1:
                    raise Exception('Data must be 1-dimensional')
                else:
                    y = y[:, 0]
            else:
                if n > 1:
                   raise Exception('Data must be 1-dimensional')
                else:
                    y = y[0, :]

        self.bokeh = bokeh_pypls(**kwargs).scatter(x, y)
        self.matplotlib = mplt_pypls(**kwargs).scatter(x, y)

    def show(self):
        show(self.bokeh)

    def save_as(self, filename):
        if isinstance(filename, str):
            self.matplotlib.savefig(filename)


def ROCline(specificity, sensitivity):

        p = figure(plot_width=600,
                   plot_height=500,
                   title="ROC",
                   x_axis_label='1-Specificity',
                   y_axis_label='Sensitivity',
                   tools="save")
        p.line(1 - specificity, sensitivity, size=10, alpha=1.0)
        return p


def ROC(pls_object, classes, positive, npoints=50):
    specificity, sensitivity = ROCanalysis(pls_object.Yhat, classes, positive, npoints=npoints)
    return ROCline(specificity, sensitivity)


def plotScores(lv_object, cp1, cp2, groups=None, labels=None, title="", save_as=None):

    # TODO: Check dimension of inputs

    TOOLS = "box_zoom,reset,hover,resize"

    T1 = lv_object.scores(cp1)
    T2 = lv_object.scores(cp2)
    if not title:
        title = "Score Plot"

    if T1 is not None and T2 is not None:
        return scatterPlot(T1,
                           T2,
                           groups=groups,
                           labels=labels,
                           title=title,
                           save_as=save_as,
                           xlabel="T["+str(cp1)+"]",
                           ylabel="T["+str(cp2)+"]",
                           hotteling=True)
    else:
        return None

def plotLoadings(lv_object, cp1, cp2, groups=None, labels=None, title=""):

    # TODO: Check dimension of inputs

    TOOLS = "box_zoom,reset,hover"

    P1 = lv_object.loadings(cp1)
    P2 = lv_object.loadings(cp2)

    if not title:
        title = "Loading Plot"

    if P1 is not None and P2 is not None:

        return scatterPlot(P1,
                           P2,
                           groups=groups,
                           labels=labels,
                           title=title,
                           xlabel="P["+str(cp1)+"]",
                           ylabel="P["+str(cp2)+"]")
    else:
        return None




def scatterYhat(pls_object, Y=None, groups=None, labels=None, cv=False, TOOLS=None, title=""):
    """
    Plot Yhat according to Y or according to a random variable to emulate a y-axis
    :param groups:
    :param labels:
    :param Yhat:
    :return:
    """
    if cv:
        Yhat = pls_object.Yhatcv
    else:
        Yhat = pls_object.Yhat
    # TODO: Check dimension of inputs and if PLS model

    if not title:
        title="Prediction of Y"
    if Y is not None:
        return scatterPlot(Y,
                           Yhat,
                           groups=groups,
                           labels=labels,
                           title=title,
                           xlabel="Y",
                           ylabel="Yhat",
                           TOOLS=TOOLS)
    else:
        try:
            n, p = Yhat.shape
            Yhat = Yhat[:, 0]
        except ValueError:
            n, = Yhat.shape
            Yhat = Yhat

        scattered_yaxis = np.random.rand(n)
        return scatterPlot(Yhat,
                           scattered_yaxis,
                           groups=groups,
                           labels=labels,
                           title=title,
                           xlabel="Yhat",
                           ylabel="Random value",
                           TOOLS=TOOLS)



