# Imports
import os
import numpy as np
import pandas as pd
import seaborn as sns
import lime
import lime.lime_tabular
from sklearn.utils import check_random_state
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt


class Lice():
    """
    Class for generating LICE plots. Static class.
    contains static methods for internal processing and produces the main output through plot_lice.   
    Can return both a full figure plot, or a dataframe containing the information of the plot. The figure can be 
    reconstructed from the dataframe if needed.
    """
    
    @staticmethod
    def par_dep(df, model, col, pcntile=0.5, pred_class=0, xbins=None):
        """
        Calculates partial dependency. For a given range, calculates the output of the model
        when only a particular feature "col" is changed according to that range. The function then calculates
        the percentile value of the output of the model over the full given dataset "df"
        
        Inputs
        ------
        df: data used to train the model
        model: the model to be analyzed
        col: feature of interest
        pcntile: percentile to compute
        pred_class: class to be predicted as 1 (In our case, we are looking at prob of default, so 1 is defautl)
        xbins: values of x to compute percentiles for
        
        Outputs
        -------
        output: numpy array of size (len(xbins),2) with the values of array xbins in first col, 
        and the percentile values of the probs for the full dataset replacing "col" by "x" at each observation.
        
        """
        
        
        
        # Prepare default xbins if none was specified
        if xbins is None:
            min_x = min(df[col])
            max_x = max(df[col])
            xbins = np.linspace(min_x, max_x, 15)

        # init empty np array that will hold
        output = np.zeros((len(xbins), 2))

        df_local = df.copy(deep=True)

        for i, x in enumerate(xbins):
            # Force each observation in the dataset to have this x
            df_local[col] = x
            output[i, :] = [x, np.percentile(model.predict_proba(df_local)[:, pred_class], int(pcntile))]

        return output

    @staticmethod
    def get_lice_bins(df, col, xres=15, yres=11):
        """
        Compute the values of the x coordinates that will be shown on the LICE plot.
        
        Inputs
        ------
        df: data used to train the model
        col: feature of interest
        xres: resolution in x axis (number of bins)
        yres: resolution in y axis (number of percentile bins)
        
        Outputs
        -------
        output: numpy array of size (len(xbins),2) with the values of array xbins in first col, 
        and the percentile values of the probs for the full dataset replacing "col" by "x" at each observation.
               
        """
        
        
        # Input checking/override
        if df[col].dropna().value_counts().index.isin([0, 1]).all():  # If a binary feature has been specified...
            xres = 2  # ...force xres to 2 regardless of specified parameter
        elif df[col].apply(float.is_integer).all():  # If the column is only integer values...
            xres = int(max(df[col]) - min(df[col])) + 1  # ... force xres to the ints regardless of specified parameter

        xbins = np.linspace(min(df[col]),
                            max(df[col]),
                            xres)
        pcntile_bins = np.linspace(0, 100, yres)
        pcntile_bins[-1] = 99  # Use the 99th percentile for the final bin instead of 100th

        return xbins, pcntile_bins

    @staticmethod
    def get_quantiles(df, model, xbins, quantile_bins, col, pred_class):
        """
        Calculates the quantiles of the data. 
        
        Inputs
        ------
        df: data used to train the model
        model: model to analyze. Follows sklearn interface.
        xbins: values of x coordinates to be shown in the plot.
        quantile_bins: values of y coordinates to be shown in the plot
        col: feature to analyze
        pred_class: class to be predicted as 1 (In our case, we are looking at prob of default, so 1 is defautl)
        
        Outputs
        -------
        df_out: dataframe with quantiles
        
        """
        
        
        # Populate output df with "what if" quantiles a la ICE
        df_sorted = df.copy(deep=True).sort_values(col).reset_index(drop=True)

        df_out = pd.DataFrame()
        df_out[col] = xbins

        for i, this_bin in enumerate(quantile_bins):
            quantile_par_dep = Lice.par_dep(df=df_sorted,
                                           model=model,
                                           col=col,
                                           pcntile=this_bin,
                                           xbins=xbins,
                                           pred_class=pred_class)
            df_out[str(int(round(this_bin, 0)))] = quantile_par_dep[:, 1]
        return df_out

    @staticmethod
    def plot_lice(df, model, col, observation, xres=18, yres=11, pred_class=0,
                  centered=False, closeness_alphaplot=True, quantile_colorplot=False, plot=True):
        """
        Main function to plot the LICE figures. The LICE plot works as a what if analysis. Given an observation and a feature, 
        the plot shows what would happen with the output of the model (prob of default) if the feature increases or decreases.
        The plot gives insights on quantiles of the data as well, to compare to the rest of the dataset.
        
        Inputs
        ------
        df: the data, as a pandas dataframe
        model: the model itself
        col: the feature we want to observe
        observation: the current observation for which to plot the what if analysis
        xres: Resolution of x axis
        yres: Resolution of y axis
        pred_class: class to be predicted as 1 (In our case, we are looking at prob of default, so 1 is default)
        centered: whether to center or not
        closeness_alphaplot: wether to fade increasing quantile curves
        quantile_colorplot: whether to plot the evolution of the quantiles for the full data as a colormap or as a fixed color
        plot: if True, builds the plot and returns the figure. If False, the function will return a dataframe with the necessary info to build the plot
        
        Returns
        -------
        if plot:
            fig: the figure containing the LICE plot.
        else:
            df_out: dataframe with plot values, plot can be reconstructed from it.
        """
        
        # Obtaining bins, values of x and y axis to be shown
        xbins, pcntile_bins = Lice.get_lice_bins(df, col, xres, yres)

        # Initialize output dataframe
        df_out = pd.DataFrame()
        df_out[col] = xbins

        # Round our observation's real x to the closest x value in our grid
        observation_x = df.loc[observation,col]
        observation_i = np.abs(xbins - observation_x).argmin()
        observation_x_rounded = xbins[observation_i]

        # Predict observation at each feature value
        df_observation_whatif = df.loc[[observation]*len(xbins),:].reset_index(drop=True)
        df_observation_whatif.loc[:, col] = xbins
        df_out['observation'] = model.predict_proba(df_observation_whatif)[:, pred_class]

        # Merge in the quantile data
        df_out = pd.merge(df_out,
                          Lice.get_quantiles(df, model, xbins, pcntile_bins, col, pred_class),
                          on=col)

        # Find our current observation's baseline quantile for visualizing "closeness" on the plot
        baseline_x = observation_x_rounded
        observation_baseline = df_out.loc[df_out[col] == baseline_x, 'observation'].values[0]
        baseline_scale_max = max(df_out.loc[df_out[col] == baseline_x, '0':'99'].values[0])
        baseline_scale_min = min(df_out.loc[df_out[col] == baseline_x, '0':'99'].values[0])
        baseline_scale_factor = max(baseline_scale_max - observation_baseline,
                                    observation_baseline - baseline_scale_min)

        if centered:
            # Adjust all of the quantiles to center around observation's true x
            for y_col in df_out.drop([col, 'observation'], axis=1):
                column_adjust = observation_baseline - df_out.loc[df_out[col] == baseline_x, y_col].values[0]
                df_out[y_col] += column_adjust

        if plot:
            y_plot_min, y_plot_max = -0.25, 1.5

            fig, ax = plt.subplots(figsize=(16, 6))
            for y_col in df_out.drop([col], axis=1):
                if y_col == 'observation':
                    # Main observation line
                    ax.plot(df_out[col], df_out[y_col],
                            color='yellow',
                            alpha=0.5,
                            linewidth=8,
                            zorder=10)
                    ax.plot(df_out[col], df_out[y_col],
                            color='black',
                            alpha=1,
                            linewidth=2,
                            marker='o',
                            markersize=5,
                            zorder=10)
                    # Vertical observation line
                    ax.axvline(x=observation_x_rounded,
                               ymin=(0 - y_plot_min) / (y_plot_max - y_plot_min),
                               ymax=(1 - y_plot_min) / (y_plot_max - y_plot_min),
                               color='yellow',
                               alpha=0.5,
                               linewidth=4,
                               zorder=10)
                    ax.axvline(x=observation_x_rounded,
                               ymin=(0 - y_plot_min) / (y_plot_max - y_plot_min),
                               ymax=(1 - y_plot_min) / (y_plot_max - y_plot_min),
                               color='black',
                               linewidth=2,
                               linestyle=':',
                               zorder=10)
                else:
                    # One of the background 'quantile' lines
                    distance_from_observation = abs(
                        observation_baseline - df_out.loc[observation_i, y_col]) / baseline_scale_factor
                    ax.plot(df_out[col], df_out[y_col],
                            color=plt.cm.ocean(0.7 * (1 - int(y_col) / 100.)) if quantile_colorplot else '#1A4E5D',
                            alpha=max(1 - distance_from_observation, 0.25) if closeness_alphaplot else 0.5,
                            linewidth=1)

                    # Locate and add quantile labels
                    x, y = df_out[col].values, df_out[y_col].values
                    label_x = (x.max() - x.min())*int(y_col)/100 + x.min()
                    label_y = np.interp(label_x, x, y)

                    plt.text(label_x, label_y, y_col, size=9, rotation=0, color='grey',
                                    ha="center", va="center",
                                    bbox=dict(ec=ax.get_facecolor(), fc=ax.get_facecolor(), pad=0))

            # Set axis parameters
            ax.set_xticks(np.linspace(min(df_out[col]), max(df_out[col]), xres))
            ax.set_yticks(np.linspace(0, 1, 5))
            ax.set_ylim((y_plot_min, y_plot_max))
#            ax.set_facecolor('white')
            ax.grid(color='grey', alpha=0.25)
            ax.set_xlabel(col)
            ax.set_ylabel('Probability of default')
            fig.text(.165, .8, "LICE Plot", fontsize=22, color='black')
            fig.text(.165, .75, f"Observation number {observation}", fontsize=18, color='grey')
            return fig
        return df_out
