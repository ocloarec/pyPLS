from matplotlib import pyplot as plt
import pandas as pd

from .plotutils import hottelingEllipse, COLOR_CODES

class mplt_pypls(object):
    def __init__(self, **kwargs):
        self.tools = ""
        self.groups = None
        self.labels = None
        self.title = ""
        self.xlabel = ""
        self.ylabel = ""
        self.hotteling = None
        self.save_as = ""

        for key, value in kwargs.iteritems():
            try:
                setattr(self, key, value)
            except AttributeError:
                pass

    def scatter(self, x, y):
        fig = plt.figure(figsize=[6,5])
        ax = fig.add_subplot(111, aspect='equal')
        plt.grid()
        xh, yh = hottelingEllipse(x, y)
        plt.plot(xh, yh, alpha=0.75, color="grey", linestyle=":")
        plt.axhline(0, color='grey', alpha=0.5)
        plt.axvline(0, color='grey', alpha=0.5)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)

        if self.groups is not None:
            df = pd.DataFrame({"x": x, "y": y, "groups": self.groups})
            df = df[["x", "y", "groups"]]

            grouped = df.groupby("groups")
            i = 0
            for name, group in grouped:
                marker_style = dict(color=COLOR_CODES[i], linestyle="", marker='o',
                                markersize=8, markerfacecoloralt='gray', alpha=0.5,
                                label=name)
                plt.plot(group["x"], group["y"], **marker_style)
                i += 1
            legend = plt.legend(numpoints = 1, framealpha=0.75)
            legend.get_frame().set_linewidth(0.25)
        else:
            marker_style = dict(color=COLOR_CODES[0], linestyle="", marker='o',
                                markersize=8, markerfacecoloralt='gray', alpha=0.5)
            plt.plot(x, y, **marker_style)


        return plt
