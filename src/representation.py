# Author: Juan Parras & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 06/08/2025

# Package imports
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.projections.polar import PolarAxes
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.projections import register_projection


def radar_factory(num_vars, frame='circle'):  # Adapted from https://stackoverflow.com/questions/52910187/how-to-make-a-polygon-radar-spider-chart-in-python
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'

        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels, fontsize=14):
            self.set_thetagrids(np.degrees(theta), labels, fontsize=fontsize)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)


                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def plot_sorted_variances(x_train, x_test, binary, delta_train, delta_test, n_logits, args, dataset):
    variances_train = [d.var(axis=0) for d in delta_train]
    variances_test = [d.var(axis=0) for d in delta_test]
    for i in range(n_logits):
        plt.figure(figsize=(8, 9))
        idxs_train = np.argsort(variances_train[i])[::-1]  # Sort by training variance

        # plt.plot(variances_train[i].values[idxs_train], label=f'train')
        # plt.plot(variances_test[i].values[idxs_train], label=f'test')

        labels = variances_train[i].index[idxs_train]
        x = np.arange(len(labels))

        # Barras para train y test (desplazadas para no solaparse)
        width = 0.4
        plt.bar(x - width / 2, variances_train[i].values[idxs_train], width=width, label='Train')
        plt.bar(x + width / 2, variances_test[i].values[idxs_train], width=width, label='Test')


        plt.title(f'Variance of delta values for logit {i}', fontsize=22)
        plt.xlabel('Feature', fontsize=20)
        plt.ylabel('Variance', fontsize=20)
        plt.legend(loc='best', fontsize=18)
        # Add the feature names
        plt.xticks(ticks=np.arange(len(variances_train[i])),
                   labels=variances_train[i].index[idxs_train],
                   rotation=90, fontsize=18)
        plt.yticks(fontsize=18)
        # Ensure that ticks are not cut off
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(args['results_folder'], dataset, f'variances_{i}.png'), bbox_inches='tight', dpi=600)
        plt.close()

    for i in range(n_logits):
        logit = 1 if binary else i
        for feat in x_train.columns:
            if variances_train[i][feat] > 1e-6:  # Only plot the features with a variance above a threshold
                plt.scatter(x_train[feat], delta_train[i][feat], label=f"{feat}_{logit}_train")
                plt.scatter(x_test[feat], delta_test[i][feat], label=f"{feat}_{logit}_test")

                # Plot the average delta values as well
                plt.plot(x_train[feat].unique(), delta_train[i][feat].mean() * np.ones_like(x_train[feat].unique()),
                         color='b', linestyle='-')
                plt.plot(x_test[feat].unique(), delta_test[i][feat].mean() * np.ones_like(x_test[feat].unique()),
                         color='r', linestyle='-')

                plt.title(f'Delta for {feat} and logit {logit}', fontsize=22)
                plt.xlabel(feat, fontsize=20)
                plt.ylabel('Delta', fontsize=20)
                plt.legend(loc='best', fontsize=18)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.savefig(os.path.join(args['results_folder'], dataset, f'delta_{feat}_logit{logit}.png'),
                            bbox_inches='tight', dpi=600)
                plt.close()  # Note: a higher delta means a higher risk


def plot_binary_explanation_plot(y_true, y_pred_proba, labels, threshold, outfile=None, title='Probability of positive class'):  # Added by Juan
    y_pred_proba = np.squeeze(np.array(y_pred_proba))

    # Umbralize the probabilities: the minimum probability is 0.01
    y_pred_proba = np.where(y_pred_proba < 0.01, 0.01, y_pred_proba)
    y_true = np.squeeze(np.array(y_true))
    fig, ax = plt.subplots()
    sort_idx = np.argsort(y_pred_proba)[::-1]  # Sort the patients by probability

    # Plot a bar diagram of the probability of each patient, where the color bar depends on the y_true value
    color_vals = [['r', 'g'][int(y)] for y in y_true[sort_idx]]
    ax.bar(range(len(y_pred_proba)), y_pred_proba[sort_idx], color=color_vals)
    ax.axhline(threshold, color='k', linestyle='--', label='Threshold')
    ax.set_yscale('log')  # Plot in log scale the vertical axis for better visualization
    ax.set_xlabel('Patient', fontsize=20)
    ax.set_ylabel('Log-Probability', fontsize=20)
    ax.set_title(title, fontsize=22)
    ax.tick_params(axis='both', labelsize=18)

    # Add the legent: red for the first label, green for the second label
    red_patch = mpatches.Patch(color='red', label=labels[0])
    green_patch = mpatches.Patch(color='green', label=labels[1])
    ax.legend(handles=[red_patch, green_patch], loc='best', fontsize=18)
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile + '_explanation.png', bbox_inches='tight', dpi=600)
    plt.close()