import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import arviz as az

# Read in the data
df = pd.read_csv("samples.csv", index_col=False)  ## Your code here
n = df.shape[0]  ## Your code here

# When is low tide?
starttime = df.iloc[:, 0][0]  ## Your code here

# Make seconds from lowtide using timestamps
## Your code here
seconds = []
for a in range(0, n):
    starttime = df.iloc[:, 0][0]
    currenttime = df.iloc[:, 0][a]
    starttime[-5:]
    currenttime[-5:]
    starthr = int(starttime[-5:][:2])
    startmin = int(starttime[-5:][-2:])
    curhr = int(currenttime[-5:][:2])
    curmin = int(currenttime[-5:][-2:])
    hr_calc = curhr - starthr
    min_calc = curmin - startmin
    if min_calc == 0:
        second_calc = 0
    else:
        second_calc = round(((hr_calc + (min_calc / 60)) * 3600), 1)
    seconds.append(second_calc)

seconds = np.array(seconds)

# Get the fish counts as a numpy array
fish_counts = np.array(df.iloc[:, 1])  ## Your code here

# How many seconds between lowtides?
period = 12.0 * 60.0 * 60.0

# Create a model
basic_model = pm.Model()  ## Your code here
with basic_model:

    # Give priors for unknown model parameters
    magnitude = pm.Uniform("magnitude", lower=0, upper=200)  ## Your code here
    sigma = pm.HalfNormal("sigma", sigma=12)  ## Your code here

    expected_count = magnitude * pm.math.sin(0.000145444 * seconds)

    target = pm.Normal("target", mu=expected_count, sigma=sigma, observed=fish_counts)

    # Create the model
    ## Your code here
    map = pm.find_MAP()

    # Make chains
    trace = pm.sample(2000, tune=500, cores=1)  ## Your code here

    # Find maximum a posteriori estimations
    map_magnitude = map["magnitude"]  ## Your code here
    map_sigma = map["sigma"]  ## Your code here

# Let the user know the MAP values
print(f"Based on these {n} measurements, the most likely explanation:")
print(
    f"\tWhen the current is moving fastest, {map_magnitude:.2f} jellyfish enter the bay in 15 min."
)
print(f"\tExpected residual? Normal with mean 0 and std of {map_sigma:.2f} jellyfish.")

posterior = trace["posterior"]
p_magnitude = posterior["magnitude"]
p_sigma = posterior["sigma"]
true_sigma = map_sigma
true_magnitude = map_magnitude
# Do a contour/density plot
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
## Your code here
ax = az.plot_kde(
    p_magnitude,
    p_sigma,
    hdi_probs=[0.3, 0.50, 0.8, 0.95],
    contourf_kwargs={"cmap": "Blues"},
)
ax.vlines(true_magnitude, true_sigma - 5, true_sigma + 5, linestyle="dashed")
ax.hlines(true_sigma, true_magnitude - 14, true_magnitude + 14, linestyle="dashed")
ax.set_xlabel("magnitude")
ax.set_ylabel("$\sigma$")
ax.set_title("Probability density of magnitude and $\sigma$ ")
fig.show()
fig.savefig("pdf.png")

# Plot your function and confidence against the observed data
hours = [n / 3600 for n in seconds]
expected_counts = [map_magnitude * np.sin(2 * np.pi * n / period) for n in seconds]
fig, ax = plt.subplots(figsize=(8, 6))
## Your code here
ax.plot(hours, fish_counts, "+", color="black", label="observed")
ax.plot(hours, expected_counts, color="red", linestyle="dashed", label="prediction")
ax.plot(
    hours,
    expected_counts - 2 * map_sigma,
    color="green",
    linestyle="dashed",
    label="95% confidence",
)
ax.plot(hours, expected_counts + 2 * map_sigma, color="green", linestyle="dashed")
ax.set_xlabel("Hours since low tide")
ax.set_ylabel("Jellyfish entering bay over 15 minutes")
ax.legend()
fig.show()
fig.savefig("jellyfish.png")
