import pymc3 as pm

BILLION = 1e9
MILLION = 1e6
TOTAL_SQUARES = 25

squares_counted = 4
yeast_counted = np.array([20, 21, 28, 34, 34, 31, 32, 32])
hours_since_inoc = np.array([0, 12.5, 17.5, 23, 36.5, 42.5, 48, 65])


def logistic(t, K, r, delta_t):
    return K / (1 + np.exp(-r * (t - delta_t)))


with pm.Model() as model:

    K = pm.Normal("K", mu=50 * MILLION, sd=25 * MILLION) # about 50% growth was expected
    P0 = pm.Normal("P0", mu=100 * MILLION, sd=25 * MILLION)
    r = pm.Exponential("r", lam=2.5)
    delta_t = pm.Uniform("delta_t", lower=0, upper=24) # lag phase happens in the first 24 hours

    yeast_conc = P0 + logistic(hours_since_inoc, K, r, delta_t)

    shaker1_volume = pm.Normal("shaker1 volume (mL)", mu=9.0, sd=0.05, shape=8)
    shaker2_volume = pm.Normal("shaker2 volume (mL)", mu=9.0, sd=0.05, shape=8)

    yeast_slurry_volume = pm.Normal("initial yeast slurry volume (mL)", mu=1.0, sd=0.01, shape=8)
    shaker1_to_shaker2_volume =    pm.Normal("shaker1 to shaker2 (mL)", mu=1.0, sd=0.01, shape=8)


    dilution_shaker1 = pm.Deterministic("dilution_shaker1", yeast_slurry_volume  / (yeast_slurry_volume + shaker1_volume))
    final_dilution_factor = pm.Deterministic("dilution_shaker2", dilution_shaker1 * shaker1_to_shaker2_volume / (shaker1_to_shaker2_volume + shaker2_volume))

    # the manufacturer suggests that depth of the chamber is 0.01cm ± 0.0004cm. Let's assume the worst and double the error.
    # the length of the square grid is 1mm = 0.1cm, to th volume is 0.01 * 0.1 * 0.1 = 0.0001, with error 0.1 * 0.1 * 0.0004 * 2
    volume_of_chamber = pm.Normal("volume of chamber (mL)", mu=1e-4, sd=8e-6)

    # why is Poisson justified? in my final shaker, I have yeast_conc * final_dilution_factor * shaker3_volume number of yeast
    # I remove volume_of_chamber / shaker3_volume fraction of them, hence it's a binomial with very high count, and very low probability.
    yeast_visible = pm.Poisson("cells in visible portion", mu=yeast_conc * final_dilution_factor * volume_of_chamber, shape=8)

    number_of_counted_cells = pm.Binomial("number of counted cells", yeast_visible, squares_counted/TOTAL_SQUARES, observed=yeast_counted, shape=8)

    trace = pm.sample(2000, tune=20000)


plt.style.use('bmh')

pm.summary(trace, var_names=['K', 'P0', 'r',])
pm.plot_posterior(trace, var_names=['K', 'P0', 'r', 'delta_t'])


# average posterior growth
plt.figure()
t = np.arange(0, 60)
results = []
for i in range(0, 4000, 40):
    y = trace['P0'][i] + logistic(t,
         trace['K'][i],
         trace['r'][i],
         trace['delta_t'][i],
         )

    results.append(y)
    plot(t, y,  color="k", alpha=0.07)

mean = np.asarray(results).mean(0)
plot(t, mean, color="k", lw=3, label="mean posterior growth curve")
plot(t, mean - np.asarray(results).std(0), color="k", lw=2, ls="--", label="1 std error")
plot(t, mean + np.asarray(results).std(0), color="k", lw=2, ls="--")
plt.title("Averaging over all posterior\nlogistic growth curves")
plt.xlabel("hours after inoculation")
plt.ylabel("yeast / mL")
plt.tight_layout()

# possible realizations
plt.figure()
for i in range(0, 4000, 1000):
    y = trace['P0'][i] + logistic(t,
         trace['K'][i],
         trace['r'][i],
         trace['delta_t'][i],
         )

    results.append(y)
    plot(t, y)
plt.title("Sampling logistic growth\ncurves from the posterior")
plt.xlabel("hours after inoculation")
plt.ylabel("yeast / mL")
plt.tight_layout()


