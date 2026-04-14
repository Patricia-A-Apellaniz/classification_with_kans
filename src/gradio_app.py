import os
import sys
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))

from data import load_data
from utils import get_config
from models.models import Kan_model
from representation import radar_factory
from copy import deepcopy
from kan import ex_round
import sympy

# ----- Data and model load -----

# Load data
print("Loading model and data...")
args = get_config('explainability')
args['datasets'] = ['heart']
x_train, x_test, y_train, y_test = load_data(args['datasets'][0], args)

raw_heart_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'ui_normalization', '7_data.csv')).drop(columns=['id'])

UI_NORMALIZED_COLS = ['Age', 'GenHlth', 'PhysHlth']
UI_NORMALIZATION_STATS = {
    col: {
        'mean': float(np.mean(raw_heart_df[col].to_numpy(dtype=float))),
        'std': float(np.std(raw_heart_df[col].to_numpy(dtype=float))),
    }
    for col in UI_NORMALIZED_COLS
}

# ----- UI code maps -----

AGE_CODE_TO_TEXT = {
    1: '18-24 years',
    2: '25-29 years',
    3: '30-34 years',
    4: '35-39 years',
    5: '40-44 years',
    6: '45-49 years',
    7: '50-54 years',
    8: '55-59 years',
    9: '60-64 years',
    10: '65-69 years',
    11: '70-74 years',
    12: '75-79 years',
    13: '80 or older',
}

GENHLTH_CODE_TO_TEXT = {
    1: 'Excellent',
    2: 'Very good',
    3: 'Good',
    4: 'Fair',
    5: 'Poor',
}

YES_NO_CODE_TO_TEXT = {
    0: 'No',
    1: 'Yes',
}

SEX_CODE_TO_TEXT = {
    0: 'Female',
    1: 'Male',
}

DIABETES_CODE_TO_TEXT = {
    0: 'No',
    1: 'Pre-diabetes',
    2: 'Yes',
}



# ----- UI mapping helpers -----

def coded_label(code, text):
    return str(text)


def format_code_for_display(col, code):
    code_int = int(code)
    if col == 'Age' and code_int in AGE_CODE_TO_TEXT:
        return AGE_CODE_TO_TEXT[code_int]
    if col == 'GenHlth' and code_int in GENHLTH_CODE_TO_TEXT:
        return GENHLTH_CODE_TO_TEXT[code_int]
    if col == 'Sex' and code_int in SEX_CODE_TO_TEXT:
        return SEX_CODE_TO_TEXT[code_int]
    if col in ['HighChol', 'DiffWalk', 'Smoker', 'Stroke'] and code_int in YES_NO_CODE_TO_TEXT:
        return YES_NO_CODE_TO_TEXT[code_int]
    if col == 'Diabetes' and code_int in DIABETES_CODE_TO_TEXT:
        return DIABETES_CODE_TO_TEXT[code_int]
    return str(code_int)


def raw_to_model_space(col, raw_value):
    stats = UI_NORMALIZATION_STATS[col]
    return (float(raw_value) - stats['mean']) / stats['std']


def model_to_raw_space(col, model_value):
    stats = UI_NORMALIZATION_STATS[col]
    return float(model_value) * stats['std'] + stats['mean']


def decode_feature_value(col, value):
    if pd.isna(value):
        return value

    if col == 'Age':
        age_group = int(round(model_to_raw_space('Age', value)))
        age_group = max(min(age_group, 13), 1)
        return format_code_for_display('Age', age_group)

    if col == 'GenHlth':
        gen_hlth = int(round(model_to_raw_space('GenHlth', value)))
        gen_hlth = max(min(gen_hlth, 5), 1)
        return format_code_for_display('GenHlth', gen_hlth)

    if col == 'PhysHlth':
        days = int(round(model_to_raw_space('PhysHlth', value)))
        days = max(min(days, 30), 0)
        return f"{days} days"

    if col in ['Sex', 'HighChol', 'DiffWalk', 'Smoker', 'Stroke']:
        code = int(round(float(value)))
        return format_code_for_display(col, code)

    if col == 'Diabetes':
        code = int(round(float(value)))
        return format_code_for_display('Diabetes', code)

    return round(float(value), 3) if isinstance(value, (float, np.floating)) else value


def decode_table_for_display(df):
    decoded = df.copy()
    for col in decoded.columns:
        if col in ['Patient', 'Train ID', 'Predicted Risk', 'Real Label']:
            continue
        decoded[col] = decoded[col].map(lambda v: decode_feature_value(col, v))

    if 'Predicted Risk' in decoded.columns:
        decoded['Predicted Risk'] = decoded['Predicted Risk'].map(lambda x: round(float(x), 3) if isinstance(x, (int, float, np.floating)) else x)
    if 'Real Label' in decoded.columns:
        decoded['Real Label'] = decoded['Real Label'].map(lambda x: int(round(float(x))) if isinstance(x, (int, float, np.floating)) else x)

    return decoded


def get_raw_ticks_for_feature(feature_name):
    if feature_name == 'Age':
        tick_codes = list(range(1, 14))
        ticks = [raw_to_model_space('Age', code) for code in tick_codes]
        labels = [format_code_for_display('Age', code) for code in tick_codes]
        return ticks, labels

    if feature_name == 'GenHlth':
        tick_codes = [1, 2, 3, 4, 5]
        ticks = [raw_to_model_space('GenHlth', code) for code in tick_codes]
        labels = [format_code_for_display('GenHlth', code) for code in tick_codes]
        return ticks, labels

    if feature_name == 'PhysHlth':
        tick_days = list(range(0, 31, 5))
        ticks = [raw_to_model_space('PhysHlth', day) for day in tick_days]
        labels = [f"{day} days" for day in tick_days]
        return ticks, labels

    if feature_name in ['Sex', 'HighChol', 'DiffWalk', 'Smoker', 'Stroke']:
        tick_codes = [0, 1]
        ticks = tick_codes
        labels = [format_code_for_display(feature_name, code) for code in tick_codes]
        return ticks, labels

    if feature_name == 'Diabetes':
        tick_codes = [0, 1, 2]
        ticks = tick_codes
        labels = [format_code_for_display('Diabetes', code) for code in tick_codes]
        return ticks, labels

    return None, None

# ----- Train model and decomposition -----

# Train KAN model
print("Training KAAM model...")
model = Kan_model(hidden_dim=0, grid=3, k=5, lamb=0.01, lamb_entropy=0.1, 
                  lr=0.001, weight=True, sparse_init=False, mult_kan=False, 
                  seed=0, batch_size=-1, steps=10000, early_stop=True)
model.run_model(x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy())

# Compute delta matrix
model.model(model.dataset['train_input'])
model.model.auto_symbolic()
var_symbols = [sympy.Symbol(f'x_{i + 1}') for i in range(x_train.shape[1])]
formula = model.model.symbolic_formula(var=var_symbols)[0]
n_digit = 2
formula = [ex_round(f, n_digit) for f in formula]
logit_formula = formula[1] - formula[0]

for i, col in enumerate(x_train.columns):
    logit_formula = logit_formula.subs(sympy.symbols(f'x_{i + 1}'), sympy.symbols(col))

# Compute delta matrices
delta_train = np.zeros((x_train.shape[0], x_train.shape[1] + 1))
for i in range(x_train.shape[0]):
    for fs in logit_formula.args:
        formula_sum_term = deepcopy(fs)
        if isinstance(formula_sum_term, sympy.Float):
            delta_train[i, -1] = float(formula_sum_term)
        else:
            variable_in_the_expresion = list(formula_sum_term.free_symbols)[0]
            variable_index = x_train.columns.get_loc(str(variable_in_the_expresion))
            delta_train[i, variable_index] += float(formula_sum_term.subs(variable_in_the_expresion, 
                                                                           x_train.iloc[i, variable_index]))

delta_test = np.zeros((x_test.shape[0], x_test.shape[1] + 1))
for i in range(x_test.shape[0]):
    for fs in logit_formula.args:
        formula_sum_term = deepcopy(fs)
        if isinstance(formula_sum_term, sympy.Float):
            delta_test[i, -1] = float(formula_sum_term)
        else:
            variable_in_the_expresion = list(formula_sum_term.free_symbols)[0]
            variable_index = x_test.columns.get_loc(str(variable_in_the_expresion))
            delta_test[i, variable_index] += float(formula_sum_term.subs(variable_in_the_expresion, 
                                                                          x_test.iloc[i, variable_index]))

variable_names = x_train.columns.tolist()


# ----- Table and plots -----

def compute_patient_delta(x_patient_df):
    cols_with_constant = variable_names + ['Constant']
    delta_patient = pd.DataFrame(np.zeros((1, len(cols_with_constant)), dtype=np.float32), columns=cols_with_constant)

    for fs in logit_formula.args:
        formula_sum_term = deepcopy(fs)
        if isinstance(formula_sum_term, sympy.Float):
            delta_patient.at[0, 'Constant'] = float(formula_sum_term)
        else:
            variable_in_the_expresion = list(formula_sum_term.free_symbols)[0]
            variable_name = str(variable_in_the_expresion)
            variable_value = float(x_patient_df.at[0, variable_name])
            delta_patient.at[0, variable_name] += float(formula_sum_term.subs(variable_in_the_expresion, variable_value))

    return delta_patient

def get_closest_patients(tr_x, tr_y, tr_d, patient_d, index_feats, patient_inf, n_closest):
    dists = np.linalg.norm(tr_d - patient_d.values, axis=1)
    idx_closest = np.argsort(dists)[:n_closest].tolist()
    pred_prob = (1 / (1 + np.exp(-tr_d[idx_closest].sum(axis=1))))
    real_label = tr_y.iloc[idx_closest].values
    closest_data = tr_x.iloc[idx_closest].values[:, index_feats]
    closest_data = np.concatenate((closest_data, pred_prob[:, None], real_label[:, None]), axis=1)
    closest_data = np.vstack((patient_inf[None, :], closest_data))
    new_df = pd.DataFrame(closest_data, columns=np.array(tr_x.columns.tolist())[index_feats].tolist() + ['Predicted Risk', 'Real Label'])
    new_df = new_df.map(lambda x: round(x, 3) if isinstance(x, float) else x)
    patient_labels = ['Your patient'] + ['Similar patient' for _ in range(len(idx_closest))]
    train_ids = ['-'] + [str(j) for j in idx_closest]
    new_df.insert(0, 'Patient', patient_labels)
    new_df.insert(1, 'Train ID', train_ids)
    new_df = new_df.reset_index(drop=True)
    return decode_table_for_display(new_df), idx_closest

def get_radar_plot(tr_d, patient_d, patient_pred_prob, n_feats, idx_vars, cols_vars, y, n_closest, closest_idx, patient_title, c):
    theta = radar_factory(n_feats, frame='polygon')
    cohort_avg_prob = 1 / (1 + np.exp(-tr_d.mean(axis=0).sum())) * np.ones(len(cols_vars))
    title = f"Risk: {patient_pred_prob:.1%} | Cohort avg: {cohort_avg_prob[0]:.1%}"

    fig, ax = plt.subplots(subplot_kw=dict(projection='radar'), figsize=(6, 6))
    ax.set_title(title, fontsize=12, pad=20)
    _ = ax.plot(theta, cohort_avg_prob, label='Cohort Avg', color=c[0], linewidth=0.5)
    ax.fill(theta, cohort_avg_prob, alpha=0.1, color=c[0])
    
    avg_delta = tr_d.mean(axis=0)[None, :]
    avg_matrix = np.repeat(avg_delta, tr_d.shape[1], axis=0)
    
    for j in range(min(n_closest, len(closest_idx))):
        np.fill_diagonal(avg_matrix, tr_d[closest_idx[j]])
        pat_proba = 1 / (1 + np.exp(-avg_matrix.sum(axis=1)))
        _ = ax.plot(theta, pat_proba[idx_vars], label='Similar patient', color='#2ca02c', alpha=0.9, linewidth=1.25)
        ax.fill(theta, pat_proba[idx_vars], alpha=0.1, color=c[1])
    
    np.fill_diagonal(avg_matrix, patient_d.values)
    pat_proba = 1 / (1 + np.exp(-avg_matrix.sum(axis=1)))
    _ = ax.plot(theta, pat_proba[idx_vars], label='Your patient', color=c[2], alpha=0.7, linewidth=1.15, linestyle='dashed')
    ax.fill(theta, pat_proba[idx_vars], alpha=0.1, color=c[2])
    ax.set_varlabels(cols_vars, fontsize=8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    return fig

def get_curves(tr_d, patient_d, cols, n_feats, cols_vars, n_closest, closest_idx, patient_x, patient_title, c):
    tr_d_df = pd.DataFrame(tr_d, columns=cols)
    
    fig, axs = plt.subplots(n_feats + 1, 1, figsize=(8, 2.0 * (n_feats + 1)))
    
    x_vals = np.arange(tr_d.sum(axis=1).min(), tr_d.sum(axis=1).max(), 0.01)
    theor_proba = 1 / (1 + np.exp(-x_vals))
    patient_pred_prob = float(1 / (1 + np.exp(-patient_d.values.sum())))
    
    axs[0].plot(x_vals, theor_proba, c[0], alpha=0.2, linewidth=0.8)
    for idj, feat_name in enumerate(cols_vars):
        if idj < n_feats:
            j = idj + 1
            idxs = np.unique(x_train[feat_name].values, return_index=True)[1]
            axs[j].plot(x_train[feat_name].values[idxs], tr_d_df[feat_name].values[idxs], color=c[0], linewidth=0.8)
            axs[j].scatter(x_train[feat_name].values[idxs], tr_d_df[feat_name].values[idxs], color=c[0], alpha=0.1)
            
            for jj in range(min(n_closest, len(closest_idx))):
                axs[j].scatter(x_train.iloc[closest_idx[jj]][feat_name], tr_d_df.iloc[closest_idx[jj]][feat_name],
                               color=c[1], alpha=0.8, s=30, marker='o')
            axs[j].scatter(patient_x[feat_name], patient_d[feat_name], color=c[2], s=70, marker='x')
            axs[j].set_ylabel(feat_name, fontsize=9)

            ticks, labels = get_raw_ticks_for_feature(feat_name)
            if ticks is not None:
                axs[j].set_xticks(ticks)
                if feat_name == 'Age':
                    axs[j].set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
                else:
                    axs[j].set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    
    axs[0].scatter(patient_d.values.sum(), patient_pred_prob, color=c[2], marker='x', s=70)
    axs[0].set_ylabel('Probability', fontsize=9)
    axs[0].set_title('Feature Contributions')
    
    red_patch = mpatches.Patch(color=c[2], label='Your patient')
    green_patch = mpatches.Patch(color=c[1], label='Similar patient')
    blue_patch = mpatches.Patch(color=c[0], label='Training population')
    plt.tight_layout()
    axs[-1].legend(handles=[red_patch, green_patch, blue_patch], loc='upper center',
                   bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=8)
    
    return fig

def analyze_custom_patient(patient_delta, patient_x):
    cols = x_train.columns.tolist() + ['Constant']
    training_delta = delta_train
    patient_pred_prob = 1 / (1 + np.exp(-patient_delta.values.sum()))
    y = -1

    variances = np.var(training_delta, axis=0)
    idx_vars = np.argsort(variances)[::-1]
    num_of_zero_var = (variances < 1e-6).sum()
    idx_vars = idx_vars[:-num_of_zero_var]
    idx_vars = np.array([i for i in idx_vars if i < len(x_train.columns)], dtype=int)
    
    max_feats = 8
    if len(idx_vars) > max_feats:
        idx_vars = idx_vars[:max_feats]

    patient_info = np.concatenate((patient_x.iloc[idx_vars].values, [patient_pred_prob, y]))
    closest_patients_df, closest_patients_idx = get_closest_patients(
        x_train, y_train, training_delta, patient_delta, idx_vars, patient_info, 5
    )
    
    # Plot the radar chart (show only high-variance features)
    idx_vars_plot = idx_vars[::-1]
    cols_vars = [cols[i] for i in idx_vars_plot.tolist()]
    n_feats = min(max_feats, len(cols_vars))
    
    fig_radar = None
    if n_feats >= 3:
        fig_radar = get_radar_plot(training_delta, patient_delta, patient_pred_prob, n_feats, idx_vars_plot,
                                   cols_vars, y, 5, closest_patients_idx, 'interactive_patient', ['b', 'g', 'r'])
    
    # Curves plot: show only the ones that do matter!!
    idx_vars_plot = idx_vars_plot[::-1]
    cols_vars = [cols[i] for i in idx_vars_plot.tolist()]
    n_feats = min(len(cols_vars), 7)
    
    fig_curves = None
    if n_feats > 0:
        fig_curves = get_curves(training_delta, patient_delta, cols, n_feats, cols_vars, 5,
                                closest_patients_idx, patient_x, 'interactive_patient', ['b', 'g', 'r'])
    
    cohort_risk = 1 / (1 + np.exp(-training_delta.mean(axis=0).sum()))
    message = (
        "<div style='padding:10px 12px; border:1px solid #9ca3af; border-radius:8px;'>"
        "<div style='font-size:16px; margin-bottom:4px;'>Predicted Risk</div>"
        f"<div style='font-size:34px; font-weight:700; line-height:1.05; margin-bottom:10px;'>{patient_pred_prob:.1%}</div>"
        "<div style='font-size:15px; margin-bottom:4px;'>Cohort Average Risk</div>"
        f"<div style='font-size:26px; font-weight:600; line-height:1.05; margin-bottom:8px;'>{cohort_risk:.1%}</div>"
        f"<div style='font-size:14px;'>Similar Patients: {len(closest_patients_idx)}</div>"
        "</div>"
    )
    
    return message, fig_radar, closest_patients_df, fig_curves


# ----- Gradio app -----

def gradio_interface(age, sex, gen_hlth, high_chol, diff_walk, smoker, stroke, phys_hlth, diabetes):
    try:
        x_patient = pd.DataFrame(np.zeros((1, len(variable_names)), dtype=np.float32), columns=variable_names)
        x_patient['DiffWalk'] = diff_walk
        x_patient['HighChol'] = high_chol
        x_patient['Sex'] = sex
        x_patient['Smoker'] = smoker
        x_patient['Stroke'] = stroke
        x_patient['Diabetes'] = diabetes
        x_patient['PhysHlth'] = raw_to_model_space('PhysHlth', phys_hlth)
        x_patient['GenHlth'] = raw_to_model_space('GenHlth', gen_hlth)
        x_patient['Age'] = raw_to_model_space('Age', age)

        delta_patient = compute_patient_delta(x_patient)

        message, fig_radar, df_closest, fig_curves = analyze_custom_patient(delta_patient, x_patient.iloc[0])
        
        return message, fig_radar, df_closest, fig_curves
    
    except Exception as e:
        error_msg = f"<div style='font-size:14px;'>Error en análisis: {str(e)}</div>"
        return error_msg, None, pd.DataFrame(), None

# Create Gradio interface
with gr.Blocks(title="KAAM Heart Failure Risk Assessment") as demo:
    gr.Markdown("""
    # Interactive Heart Failure Risk Assessment

    Adjust patient characteristics below to assess heart failure risk with local explanations.
    """)
    
    with gr.Row():
        with gr.Column(scale=0.8):
            age = gr.Dropdown(
                label="Age",
                choices=[
                    (coded_label(1, '18-24 years'), 1),
                    (coded_label(2, '25-29 years'), 2),
                    (coded_label(3, '30-34 years'), 3),
                    (coded_label(4, '35-39 years'), 4),
                    (coded_label(5, '40-44 years'), 5),
                    (coded_label(6, '45-49 years'), 6),
                    (coded_label(7, '50-54 years'), 7),
                    (coded_label(8, '55-59 years'), 8),
                    (coded_label(9, '60-64 years'), 9),
                    (coded_label(10, '65-69 years'), 10),
                    (coded_label(11, '70-74 years'), 11),
                    (coded_label(12, '75-79 years'), 12),
                    (coded_label(13, '80 or older'), 13),
                ],
                value=9,
            )
            sex = gr.Dropdown(
                label="Sex",
                choices=[(coded_label(0, 'Female'), 0), (coded_label(1, 'Male'), 1)],
                value=0,
            )
            gen_hlth = gr.Dropdown(
                label="General Health",
                choices=[
                    (coded_label(1, 'Excellent'), 1),
                    (coded_label(2, 'Very good'), 2),
                    (coded_label(3, 'Good'), 3),
                    (coded_label(4, 'Fair'), 4),
                    (coded_label(5, 'Poor'), 5),
                ],
                value=3,
            )
            high_chol = gr.Dropdown(label="High Cholesterol", choices=[(coded_label(0, 'No'), 0), (coded_label(1, 'Yes'), 1)], value=0)
            diff_walk = gr.Dropdown(label="Difficulty Walking", choices=[(coded_label(0, 'No'), 0), (coded_label(1, 'Yes'), 1)], value=0)
            smoker = gr.Dropdown(label="Smoker", choices=[(coded_label(0, 'No'), 0), (coded_label(1, 'Yes'), 1)], value=0)
            stroke = gr.Dropdown(label="Stroke History", choices=[(coded_label(0, 'No'), 0), (coded_label(1, 'Yes'), 1)], value=0)
            phys_hlth = gr.Slider(label="Number of Days Physical Health was Not Good (In Last 30 days)", value=0, minimum=0, maximum=30, step=1)
            diabetes = gr.Dropdown(
                label="Diabetes",
                choices=[
                    (coded_label(0, 'No'), 0),
                    (coded_label(1, 'Pre-diabetes'), 1),
                    (coded_label(2, 'Yes'), 2),
                ],
                value=0,
            )

        with gr.Column(scale=3):
            gr.Markdown("### Risk Assessment")
            risk_output = gr.HTML(
                value="<div style='padding:10px 12px; border:1px solid #9ca3af; border-radius:8px; font-size:16px;'>Loading risk assessment...</div>",
                show_label=False,
            )

            gr.Markdown("### Similar Patients Database")
            table_closest = gr.Dataframe(
                value=pd.DataFrame(),
                headers=None,
                row_count=0,
                interactive=False,
                scale=1,
                show_label=False,
                min_width=900,
                # Last two columns are Predicted Risk and Real Label.
                column_widths=[130, 70] + [78] * 8 + [105, 80],
            )

    with gr.Row():
        fig_radar = gr.Plot(label="Risk Profile (Radar Chart)")
    
    with gr.Row():
        fig_curves = gr.Plot(label="Feature Contributions (Curves)")

    input_components = [age, sex, gen_hlth, high_chol, diff_walk, smoker, stroke, phys_hlth, diabetes]
    output_components = [risk_output, fig_radar, table_closest, fig_curves]
    for component in input_components:
        component.change(fn=gradio_interface, inputs=input_components, outputs=output_components)

    demo.load(
        fn=gradio_interface,
        inputs=input_components,
        outputs=output_components,
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7862, share=False)
