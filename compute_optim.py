import numpy as np
import matplotlib.pyplot as plt

"""
Code for computing the optimal strategy for AFK Arena boss fights.
I know nothing about the game, I just found the problem interesting.
There is probably some bugs in computing the values, so use at your own risk.
- Rasmus Tirsgaard 17/11/2022
"""

np.random.seed(1)

# These are game specific parameters
save_cost = 12  # Cost to lock in
energy = 400-save_cost  # Total energy (-save_cost is for first sample)
N_uncertainty = 1000  # Number of times to sample to generate uncertainty. Reduce this to speed up calculation

X_data = np.array(
    [4.557, 5.145, 4.563, 4., 4.529, 5.69, 4.771, 5.81, 6.401, 5.802, 5.621, 4.546, 4.337, 5.522, 4.409, 5.196, 4.395,
        4.404, 4.7575, 4.629, 2.857, 4.594, 4.524, 4.96, 6.446, 4.484, 5.012, 4.547, 4.449, 4.103, 4.497, 4.473, 4.536,
        4.518, 4.411, 4.43, 4.492, 4.563, 5.303, 5.288, 4.685, 4.553, 4.488, 4.506, 5.843, 4.564, 4.985, 4.683, 4.516,
        3.204, 4.404, 5.064, 5.339, 5.452, 2.997, 4.775, 4.655, 4.532, 4.26, 4.655, 4.544, 5.064, 4.391, 4.251, 4.241,
        4.702, 4.641, 4.361, 5.944, 4.22, 4.666, 5.741, 4.434, 4.636, 5.675, 4.451, 4.831, 4.521, 4.252, 4.322, 5.439,
        4.425, 4.619, 5.169, 4.293, 5.967, 4.514, 4.431, 4.441, 4.559, 5.667, 5.325, 5.081, 6.645, 4.411, 4.523, 4.507,
        5.705, 4.833, 4.503, 4.422, 4.467, 4.379, 4.693, 4.429, 4.785, 4.373, 4.561, 5.27, 4.657, 5.907, 5.181, 4.551,
        6.277, 5.154, 4.609, 4.468, 5.052, 4.43, 4.388, 4.712, 5.365, 4.286, 4.541, 4.816, 5.286, 5.701, 5.117, 4.572,
        4.212, 4.624, 4.486, 4.675, 4.366, 4.641, 4.213, 6.435, 4.426, 4.455, 4.5, 3.065, 5.580, 4.422, 4.709, 4.132,
        4.728, 4.603, 4.567, 4.760, 4.444, 4.834, 4.684, 4.464, 5.228, 4.912, 4.575, 4.529, 4.495, 5.314, 5.276, 4.498,
        4.3, 4.519, 4.449, 6.401, 4.385, 4.634, 4.446, 4.582, 4.512, 4.631, 5.205, 4.373, 4.559, 4.569, 4.363, 4.051,
        4.501, 4.698, 4.647, 4.417, 4.337, 4.647, 5.368, 5.216, 4.519, 4.614, 4.568, 5.32, 4.782, 4.61, 5.272, 4.323,
        4.488, 4.401, 4.237, 4.567, 4.425, 5.044, 4.590, 5.689, 4.660, 4.415, 4.563, 5.57, 4.366, 4.41, 4.730, 4.479,
        4.497, 4.930, 4.596, 5.931, 4.614, 4.606, 4.635, 5.556, 5.069, 5.278, 5.321, 4.504, 2.960, 4.377, 4.655, 4.477,
        4.519, 4.475, 4.485, 4.584, 4.775, 4.691, 4.474, 4.565, 5.760, 4.294, 5.211, 4.440, 4.743, 4.512, 4.506, 4.416,
        5.927, 4.439, 4.556, 5.586, 4.402, 4.955, 4.422, 4.655, 4.597])

#### INSERT YOUR OWN DATA HERE AS AN ARRAY. SHOULD NOT BE ADDED TO X_data. THAT WILL BE DONE AUTOMATIC
X_own = np.array([])

if X_own.shape[0] >= 1:
    # Compute CDF dist
    min_obs = np.min([X_data.min(), X_own.min()])
    max_obs = np.max([X_data.max(), X_own.max()])
    x_range = np.linspace(min_obs, max_obs, 10000)

    n = X_data.shape[0]
    m = X_own.shape[0]
    CDF_data = (X_data[None]<x_range[:, None]).mean(axis=1)
    CDF_own = (X_own[None]<x_range[:, None]).mean(axis=1)
    # Use the two-sample Kolmogorov–Smirnov test
    D = np.abs(CDF_data-CDF_own).max()

    use_own_samples = D > 1.731*np.sqrt( (n+m)/(n*m))
    print("Using own data samples: " + str(use_own_samples))
    if use_own_samples:
        X_data = X_own
    else:
        X_data = np.concatenate([X_data, X_own])

# Some functions
def cond_ex(dist_array, cond):
    return np.mean(dist_array[dist_array>=cond])

def cdf(dist_array, x):
    return np.mean(dist_array<=x)

def sample_point(dist_array):
    return dist_array[np.random.randint(0, dist_array.shape[0])]

def calc_opt_strat(X, energy, save_cost):
    treshold_array = np.zeros(energy + 1)
    value_array = np.zeros(energy + 1, dtype=np.float64)
    ## Solve problem with dynamic programming
    # Initial conditions
    value_array[0] = X.mean()
    for k in range(1, save_cost):
        treshold_array[k] = value_array[k - 1]
        cdf_thresh = cdf(X, treshold_array[k])
        value_array[k] = cdf_thresh * value_array[k - 1] + (1 - cdf_thresh) * cond_ex(X, treshold_array[k])

    # Case where there is energy for new sample
    for k in range(save_cost, energy + 1):
        treshold_array[k] = value_array[k - 1] - value_array[k - save_cost]
        cdf_thresh = cdf(X, treshold_array[k])
        value_array[k] = cdf_thresh * value_array[k - 1] + (1 - cdf_thresh) * (
                    value_array[k - save_cost] + cond_ex(X, treshold_array[k]))
    return value_array, treshold_array

# Plot histogram
plt.hist(X_data, bins=20)
plt.show()



### Generate samples to estimate uncertainty
# Prepare large array
overall_value_array = np.zeros((N_uncertainty, energy+1))
overall_treshold_array = np.zeros((N_uncertainty, energy+1))
overall_value_array_heuristic = np.zeros((N_uncertainty, energy+1))
overall_mean = np.zeros((N_uncertainty))

for i in range(N_uncertainty):
    X = X_data[np.random.randint(0, X_data.shape[0], X_data.shape[0])] # Sample with replacement
    x_mean = X.mean()
    # Compute optimal strategy
    value_array, treshold_array = calc_opt_strat(X, energy, save_cost)

    # Get results for 11/12 strategy
    value_array_heuristic = np.zeros(energy+1, dtype=np.float64)
    value_array_heuristic[0] = x_mean
    for k in range(1, energy+1):
        cdf_thresh = cdf(X, x_mean*11/12)
        value_array_heuristic[k] = cdf_thresh*value_array_heuristic[k-1] + (1-cdf_thresh)*(value_array_heuristic[k-save_cost]+cond_ex(X, x_mean*11/12))

    # Save results
    overall_value_array[i, :] = value_array
    overall_treshold_array[i, :] = treshold_array
    overall_value_array_heuristic[i, :] = value_array_heuristic
    overall_mean[i] = x_mean

### Plot Uncertainty
# Plot uncertainty of value for optimal
value_percent = np.percentile(overall_value_array, [5, 95], axis=0)
max_val = value_percent[0].shape[0]
plt.plot(np.arange(max_val), value_percent[0], color="orange", linestyle="dashed")
plt.plot(np.arange(max_val), value_percent[1], color="orange", linestyle="dashed")
plt.plot(np.arange(max_val), overall_value_array.mean(axis=0), label="Predicted Optimal", color="orange")
# Plot uncertainty of value for heuristic
value_percent = np.percentile(overall_value_array_heuristic, [5, 95], axis=0)
plt.plot(np.arange(max_val), value_percent[0], color="blue", linestyle="dashed")
plt.plot(np.arange(max_val), value_percent[1], color="blue", linestyle="dashed")
plt.plot(np.arange(max_val), overall_value_array_heuristic.mean(axis=0), label="11/12 Heuristic", color="blue")
plt.xlabel("Resources left")
plt.ylabel("Reroll threshold")
plt.legend()
plt.show()

# Plot uncertainty of tresholds for optimal
treshold_percent = np.percentile(overall_treshold_array, [5, 95], axis=0)
plt.plot(np.arange(max_val), treshold_percent[0], color="orange", linestyle="dashed")
plt.plot(np.arange(max_val), treshold_percent[1], color="orange", linestyle="dashed")
plt.plot(np.arange(max_val), treshold_percent.mean(axis=0), label="Predicted Optimal", color="orange")
optimal_treshold = np.percentile(overall_treshold_array, 50, axis=0)
# Plot uncertainty of treshold for heuristic
treshold_percent = np.percentile(overall_mean, [5, 95], axis=0)
plt.plot(np.arange(max_val), treshold_array*0+treshold_percent[0]*11/12, color="blue", linestyle="dashed")
plt.plot(np.arange(max_val), treshold_array*0+treshold_percent[1]*11/12, color="blue", linestyle="dashed")
plt.plot(np.arange(max_val), treshold_array*0+overall_mean.mean(axis=0)*11/12, label="11/12 Heuristic", color="blue")
plt.xlabel("Resources left")
plt.ylabel("Average value")
plt.legend()
plt.show()


## Compute optimal strategy from non-bootstrapped data
X = X_data
value_array, optimal_treshold = calc_opt_strat(X, energy, save_cost)
x_mean = X.mean()

#### Simulate returns
N_tries = 10**5

np.random.seed(2)
# Optimal strategy
score_array = np.zeros(N_tries)
for i in range(N_tries):
    score = 0
    cur_energy = energy
    sample = sample_point(X) #np.random.rand()

    while (energy>0):
        if cur_energy>save_cost:
            # Case where energy left to sample
            if sample <= optimal_treshold[cur_energy]:
                # Reroll case
                cur_energy -= 1
                sample = sample_point(X)
            else:
                # Save sample
                score += sample
                cur_energy -= save_cost
                sample = sample_point(X)
        else:
            if sample <= optimal_treshold[cur_energy]:
                # Reroll case
                cur_energy -= 1
                sample = sample_point(X)
            else:
                # Case where we stop
                score += sample
                break
    score_array[i] = score
optimal_strat = score_array.mean()
print("Optimal strat mean value: " + str(optimal_strat))

# 11/12 strategy
np.random.seed(2)
score_array = np.zeros(N_tries)
for i in range(N_tries):
    score = 0
    cur_energy = energy
    sample = sample_point(X)

    while (energy>0):

        if cur_energy>save_cost:
            # Case where energy left to sample
            if sample <= x_mean*11/12:
                # Reroll case
                cur_energy -= 1
                sample = sample_point(X)
            else:
                # Save sample
                score += sample
                cur_energy -= save_cost
                sample = sample_point(X)
        else:
            if sample <= x_mean*11/12:
                # Reroll case
                cur_energy -= 1
                sample = sample_point(X)
            else:
                # Case where we stop
                score += sample
                break
    score_array[i] = score
mean_strat = score_array.mean()
print("11/12 mean strat value: " + str(mean_strat))

# Hybrid strategy
np.random.seed(2)
score_array = np.zeros(N_tries)
for i in range(N_tries):
    score = 0
    cur_energy = energy
    sample = sample_point(X)
    while (energy>0):

        if cur_energy>save_cost:
            # Case where energy left to sample
            if sample <= x_mean*11/12:
                # Reroll case
                cur_energy -= 1
                sample = sample_point(X)
            else:
                # Save sample
                score += sample
                cur_energy -= save_cost
                sample = sample_point(X)
        else:
            if sample <= optimal_treshold[cur_energy]:
                # Reroll case
                cur_energy -= 1
                sample = sample_point(X)
            else:
                # Case where we stop
                score += sample
                break
    score_array[i] = score
hybrid_strat = score_array.mean()
print("Hybrid strat value: " + str(hybrid_strat))

if (optimal_strat >= mean_strat) and (optimal_strat >= hybrid_strat):
    print("Optimal strategy is best")

if (mean_strat >= optimal_strat) and (mean_strat >= hybrid_strat):
    print("11/12 strategy is best")

if (hybrid_strat >= optimal_strat) and (hybrid_strat >= mean_strat):
    print("Hybrid strategy is best")

print("Hybrid strat is:")
print(optimal_treshold[0:13])
print(x_mean)

result = np.stack([np.arange(optimal_treshold.shape[0]), optimal_treshold]).transpose((1,0))
print("Reroll treshold for 1 to 12 resources")
print(result[1:(save_cost+1)])
print("Treshold above 12 resources")
print(X_data.mean())
