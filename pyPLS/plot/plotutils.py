import numpy as np
from scipy.stats import f

COLOR_CODES = ["#FF0000", "#66FF33", "#0066FF",
               "#FF9933", "#00FFCC", "#9966FF",
               "#CC6699", "#FFCC00", "#0066FF",
               "#0099CC", "#CC3300", "#9966FF",
               "#FF0000", "#66FF33", "#0066FF"]

def hottelingEllipse(T1, T2, alpha=0.95):

    H = 2
    # Initialisation
    n1 = T1.shape[0]
    n2 = T2.shape[0]

    if n1 == n2:
        hotlim = H*(n1*n1-1)/(n1*(n1-2)) * f.ppf(alpha, 2, n1-2)
        x = np.arange(0, 2*np.pi+0.01, 0.01)
        Th1 = np.sqrt(hotlim*(n1-1)/n1)*np.std(T1)*np.cos(x)
        Th2 = np.sqrt(hotlim*(n1-1)/n1)*np.std(T2)*np.sin(x)

        return Th1, Th2

    else:
        return None
