from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool
from .plotutils import hottelingEllipse, COLOR_CODES
import pandas as pd


class bokeh_pypls(object):
    def __init__(self, **kwargs):
        self.tools = ""
        self.groups = None
        self.labels = None
        self.title = ""
        self.xlabel = ""
        self.ylabel = ""
        self.hotteling = None

        for key, value in kwargs.iteritems():
            try:
                setattr(self, key, value)
            except AttributeError:
                pass

    def scatter(self, x, y):


        if not self.tools:
            TOOLS = "box_zoom,reset,save,resize"
            if self.labels is not None:
                TOOLS = "box_zoom,reset,hover,save,resize"
        else:
            TOOLS = ""

        p = figure(plot_width=600,
                   plot_height=500,
                   title=self.title,
                   x_axis_label=self.xlabel,
                   y_axis_label=self.ylabel,
                   tools=TOOLS)

        alpha = 0.5

        if self.groups is not None:
            df = pd.DataFrame({"x": x, "y": y, "groups": self.groups})
            df = df[["x", "y", "groups"]]

            grouped = df.groupby("groups")
            i = 0
            for name, group in grouped:
                p.scatter(group["x"], group["y"], size=12, legend=name, color=COLOR_CODES[i], alpha=0.5)
                i += 1
        else:
            p.scatter(x, y, size=12, color=COLOR_CODES[0], alpha=alpha)

        if self.hotteling:
            xh, yh = hottelingEllipse(x, y)
            p.line(xh, yh, alpha=0.5)

        if self.labels is not None:
            source = ColumnDataSource(
                data=dict(
                    x=x,
                    y=y,
                    label=self.labels,
                    )
            )

            p.circle(x, y, size=12,
                     fill_color=COLOR_CODES[0],
                     source=source,
                     fill_alpha=0.0,
                     line_color=None)
            hover = p.select(dict(type=HoverTool))
            hover.tooltips = [
                ("ID", "@label"),
            ]
            p.tools[2].renderers.append(p.renderers[-1])


        return p
