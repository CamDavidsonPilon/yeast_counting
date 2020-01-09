# main.py
import pymc3 as pm
import streamlit as st


BILLION = 1e9
TOTAL_SQUARES = 25


def generate_model(
        number_of_serial_dilutions,
        ml_of_each_dilution,
        ml_of_slurry_transferred,
        squares_counted,
        cells_counted,
        prior_lower_bound,
        prior_upper_bound,
        depth_of_chamber
        ):

    with pm.Model() as model:

        yeast_conc = pm.Uniform("cells/mL", lower=prior_lower_bound, upper=prior_upper_bound)


        dilution_factor = 1.
        for i in range(number_of_serial_dilutions):
            yeast_slurry_volume = pm.Normal("yeast slurry volume (mL)_%d" % i, mu=ml_of_slurry_transferred, sd=0.01)
            shaker_volume = pm.Normal("shaker volume (mL)_%d" % i, mu=ml_of_each_dilution, sd=0.05)
            dilution_factor = dilution_factor * yeast_slurry_volume / (yeast_slurry_volume + shaker_volume)

        # the manufacturer suggests that depth of the chamber is 0.01cm ± 0.0004cm. Let's assume the worst and double the error.
        # the length of the 5x5 square grid is 1mm = 0.1cm, so the volume is 0.01 * 0.1 * 0.1 = 0.0001, with error 0.1 * 0.1 * 0.0004 * 2
        volume_of_chamber = pm.Normal("volume of chamber (mL)", mu=depth_of_chamber * 0.1 * 0.1, sd=8e-6)

        # why is Poisson justified? in my final shaker, I have yeast_conc * final_dilution_factor * shaker3_volume number of yeast
        # I remove volume_of_chamber / shaker3_volume fraction of them, hence it's a binomial with very high count, and very low probability.
        yeast_visible = pm.Poisson("cells in visible portion", mu=yeast_conc * dilution_factor * volume_of_chamber * BILLION)

        number_of_counted_cells = pm.Binomial("number of counted cells", yeast_visible, squares_counted/TOTAL_SQUARES, observed=cells_counted)

        trace = pm.sample(1000, tune=2000, chains=2, init='map')

    return trace, model

st.header("(Bayesian) Cell Counting Using a Hemocytometer")
st.markdown("Read the accompanying blog post [here](https://dataorigami.net/blogs/napkin-folding/bayesian-cell-counting).")

st.sidebar.markdown('Enter your observed data here.')
squares_counted = st.sidebar.number_input("Hemocytometer squares counted", value=5)
cells_counted = st.sidebar.number_input("Cells counted", value=20)
st.sidebar.markdown('---')
st.sidebar.markdown('Below are other parameters you may wish to change.')
number_of_serial_dilutions = st.sidebar.slider("Number of dilutions", 0, 5, value=0)
ml_of_each_dilution = st.sidebar.number_input("mL of each dilution", value=9.)
ml_of_slurry_transferred = st.sidebar.number_input("mL of slurry transfer between dilutions", value=1.)
prior_lower_bound = st.sidebar.number_input("Prior conc. lower-bound (Billions)", value=0)
prior_upper_bound = st.sidebar.number_input("Prior conc. upper-bound (Billions)", value=10)
depth_of_chamber = st.sidebar.number_input("Depth of hemocytometer chamber (cm)", value=0.01)

trace, _ = generate_model(
    number_of_serial_dilutions,
    ml_of_each_dilution,
    ml_of_slurry_transferred,
    squares_counted,
    cells_counted,
    prior_lower_bound,
    prior_upper_bound,
    depth_of_chamber
)
st.write(pm.summary(trace, var_names=['cells/mL'], credible_interval=0.95).T.iloc[:4])
ax = pm.plot_posterior(trace, var_names=['cells/mL'], credible_interval=0.95, kind='hist')
ax[0].set_title("Yeast cells/mL (Billions)")
st.pyplot()


