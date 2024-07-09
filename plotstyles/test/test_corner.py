import numpy as np
from plotstyles.figure import Figure
from plotstyles.fonts import global_fonts

# Set up the parameters of the problem.
ndim, nsamples = 3, 50000

# Generate some fake data.
np.random.seed(42)
mean = np.array([1, 2, 3, 4])
cov = np.array([
    [0.6, 0, 0, 0],
    [0, 0.6, 0, 0],
    [0, 0, 1.6, 0],
    [0, 0, 0, 0.6]
])
data = np.random.multivariate_normal(mean, cov, size=10000)

fig = Figure(20, 20)
fig.corner_plot(
    data,
    bins=20,
    labels=[
        r"$x$",
        r"$y$",
        r"$z$",
        r"$w$",
    ],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
    fig = fig
)
fig.show()