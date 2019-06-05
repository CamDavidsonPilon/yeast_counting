import pymc3 as pm

BILLION = 1e9
TOTAL_SQUARES = 25

squares_counted = 4
yeast_counted = 71


with pm.Model() as model:
    yeast_conc = pm.Uniform("cells/mL", lower=0, upper=10 * BILLION)

    shaker1_volume = pm.Normal("shaker1 volume (mL)", mu=9.0, sd=0.05)
    shaker2_volume = pm.Normal("shaker2 volume (mL)", mu=9.0, sd=0.05)

    yeast_slurry_volume = pm.Normal("initial yeast slurry volume (mL)", mu=1.0, sd=0.01)
    shaker1_to_shaker2_volume =    pm.Normal("shaker1 to shaker2 (mL)", mu=1.0, sd=0.01)


    dilution_shaker1 = pm.Deterministic("dilution_shaker1", yeast_slurry_volume  / (yeast_slurry_volume + shaker1_volume))
    final_dilution_factor = pm.Deterministic("dilution_shaker2", dilution_shaker1 * shaker1_to_shaker2_volume / (shaker1_to_shaker2_volume + shaker2_volume))

    volume_of_chamber = pm.Gamma("volume of chamber (mL)", mu=0.0001, sd=0.0001 / 20)

    # why is Poisson justified? in my final shaker, I have yeast_conc * final_dilution_factor * shaker3_volume number of yeast
    # I remove volume_of_chamber / shaker3_volume fraction of them, hence it's a binomial with very high count, and very low probability.
    yeast_visible = pm.Poisson("cells in visible portion", mu=yeast_conc * final_dilution_factor * volume_of_chamber)

    number_of_counted_cells = pm.Binomial("number of counted cells", yeast_visible, squares_counted/TOTAL_SQUARES, observed=yeast_counted)

    trace = pm.sample(6000, tune=3500, nuts_kwargs=dict(target_accept=.85))


pm.summary(trace, varnames=['cells/mL'])
pm.plot_posterior(trace, varnames=['cells/mL'])

