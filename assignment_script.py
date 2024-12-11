import pandas as pd
import pymc as pm
import numpy as np
import arviz as az

# Load the dataset
url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/cookie_cats.csv"
data = pd.read_csv(url)
view_data = data.head()
print(view_data)
# Group the data by 'version' for retention1 and retention7
grouped_data = data.groupby('version').agg(
    retention1_mean=('retention_1', 'mean'),
    retention1_count=('retention_1', 'count'),
    retention7_mean=('retention_7', 'mean'),
    retention7_count=('retention_7', 'count')
).reset_index()
print(grouped_data)

# Prepare the model 
# Define Bayesian model for retention1
with pm.Model() as retention1_model:
# Priors for probabilities
 p_control = pm.Beta("p_control", alpha=1, beta=1)
 p_treatment = pm.Beta("p_treatment", alpha=1, beta=1)

# Observed data
control_data = data[data['version'] == 'gate_30']['retention_1']
treatment_data = data[data['version'] == 'gate_40']['retention_1']

obs_control = pm.Binomial("obs_control", n=len(control_data), p=p_control, observed=control_data.sum())
obs_treatment = pm.Binomial("obs_treatment", n=len(treatment_data), p=p_treatment, observed=treatment_data.sum())

# Difference in probabilities
diff_retention1 = pm.Deterministic("diff_retention_1", p_treatment - p_control)

# Sampling
retention1_trace = pm.sample(2000, return_inferencedata=True)

# Define Bayesian model for retention7
with pm.Model() as retention7_model:
# Priors for probabilities
p_control = pm.Beta("p_control", alpha=1, beta=1)
p_treatment = pm.Beta("p_treatment", alpha=1, beta=1)

# Observed data
control_data = data[data['version'] == 'gate_30']['retention_7']
treatment_data = data[data['version'] == 'gate_40']['retention_7']

obs_control = pm.Binomial("obs_control", n=len(control_data), p=p_control, observed=control_data.sum())
obs_treatment = pm.Binomial("obs_treatment", n=len(treatment_data), p=p_treatment, observed=treatment_data.sum())

# Difference in probabilities
diff_retention7 = pm.Deterministic("diff_retention_7", p_treatment - p_control)

# Sampling
retention7_trace = pm.sample(2000, return_inferencedata=True)

# Analyze and visualyze 
# Posterior summaries
print(az.summary(retention1_trace, var_names=["diff_retention_1"]))
print(az.summary(retention7_trace, var_names=["diff_retention_7"]))

# Visualize posterior distributions
az.plot_posterior(retention1_trace, var_names=["diff_retention_1"], hdi_prob=0.95)
az.plot_posterior(retention7_trace, var_names=["diff_retention_7"], hdi_prob=0.95)
