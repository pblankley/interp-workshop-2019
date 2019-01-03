# Imports
import numpy as np
import pandas as pd
import seaborn as sns
import lime
import lime.lime_tabular
from mpld3 import fig_to_html
from IPython.display import display
from tqdm import tqdm
from sklearn.utils import check_random_state
from sklearn.neighbors import BallTree
from timeit import default_timer as timer
from scipy.stats import gaussian_kde

import os
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

class InputGrad:
    def __init__(self, model, data):
        """
        Args: model; the sklearn model or any model / namedtuple with a '.predict_proba'
                function and a 'predict' function.
              data; the dataframe with the points and columns we are interested in

        NOTE: This will only take one hot vectors (like your actual model) do not put in
              categoricals with more than one class because it will treat them like quantitative variables
              just like your original model's coefficients would have.
        """
        self.model = model
        if isinstance(data, pd.DataFrame):
            self.data_mat = data.values
            self.cols = data.columns
            # Mask that is true if categorical (binary) and false if not
            self.categorical = np.array([tuple(np.unique(self.data_mat[:,i]))==(0,1) for i in range(len(self.cols))])
            self.qgrad_to_orig = [i for i,cat in enumerate(self.categorical) if not cat]
            self.cgrad_to_orig = [i for i,cat in enumerate(self.categorical) if cat]
        else:
            raise ValueError('This function does not support input types except pandas DataFrames.')

    def _finite_diff(self,x,col,h=1e-4):
        """ Finite difference of a model with respect to the specified feature.
        --------
        Args: x; numpy array, the row of the dataset we are concerned with
              col; int, the index of the column with which we are taking the derivative
                in respect to.
              h; float; the h term to be added in the finite difference formula.
        --------
        Returns: float; the result of the finite difference differention for probability
                    of default. i.e model outputs [0.1,0.5] probs
                    and we take the derivative in respect to the jth input we will
                    get a result of the form [.023,.1922] from finite difference.
                    We take the first [0] of this output because the [0] index
                    corresponds to the probability of default to get the value of
                    the partial w.r.t. the jth input of the probability of default.
        --------
        NOTE: This function assumes 0 means default and 1 means fully paid.
        """
        x = x.astype('float64').reshape(1,-1)
        x_plus_h = np.copy(x)
        x_plus_h[:,col]+=h
        prob_grad = (self.model.predict_proba(x_plus_h)-self.model.predict_proba(x))/h
        return prob_grad.flatten()[0]

    def _cat_pseudo_fd(self,x,col):
        """ Pseudo finite difference of a model with respect to the specified **categorical** feature.

        NOTE: Here I am treating the categorical pseudo gradient as the difference in the predicted
                 probabilities with a categorical (binary ONLY) flip.  The categorical changes that
                 produce the largest predicted probability changes should have larger impact on the
                 final decision.

        NOTE: this scale is almost DEFINITELY not comparable with the scale of the quantitative true
                  gradients.  Do not compare them although you might be tempted to.
        --------
        Args: x; numpy array, the row of the dataset we are concerned with
              col; int, the index of the column with which we are taking the derivative
                in respect to.
        --------
        Returns: float; the result of the finite difference differention for probability
                    of default. i.e model outputs [0.1,0.5] probs
                    and we take the derivative in respect to the jth input we will
                    get a result of the form [.023,.1922] from finite difference.
                    We take the first [0] of this output because the [0] index
                    corresponds to the probability of default to get the value of
                    the partial w.r.t. the jth input of the probability of default.
        --------
        NOTE: This function assumes 0 means default and 1 means fully paid.
        """
        x = x.astype("float64").reshape(1,-1)
        if x[:,col][0] not in {0,1}:
            raise ValueError(f'Type issues w/ categorical column you passed {x[:,col]} not in (0,1)')
        x_cat_change = np.copy(x)
        x_cat_change[:,col] = 0 if x_cat_change[:,col][0]==1 else 1
        cat_grad = self.model.predict_proba(x_cat_change)-self.model.predict_proba(x)
        return cat_grad.flatten()[0]

    def _grad(self, point, h=1e-4):
        """ Get the gradient of the model w.r.t. the specified point.
        -------
        Args: point; int, the row in our dataset we are interested in
              h; float; the term to be added in finite difference.
        -------
        Returns: tuple of numpy arrays; the gradient w.r.t the quantitative features and the
                    pseudo gradient w.r.t the categorical features of the model at the specified point
        -------
        NOTE: the input x must be a numpy ndarray.
        """
        instance = self.data_mat[point,:]
        qgrad = [self._finite_diff(instance, i, h=h) for i,cat in enumerate(self.categorical) if not cat]
        cgrad = [self._cat_pseudo_fd(instance, i) for i,cat in enumerate(self.categorical) if cat]
        return np.array(qgrad), np.array(cgrad)

    def _quant_follow_grad(self, q_col_idxs, q_col_names, qgrad, init_data, init_pred, max_itr):
        """ Helper function to find the change needed in a quantitative column to flip the decision
        boundary of the model in the class.
        -------
        Args: q_col_idxs; iterable, contains the column indexes of the quantitative columns to analyze
              q_col_names; iterable, contains the colum ns names of the quantitative columns
              qgrad; numpy array; the gradient of the quantitative columns
              init_data; numpy array, the initial row we are concerned with
              init_pred; int; (1 or 0 for binary classification) the original classification of our model
        -------
        Returns: list; the changes (in order of columns passed) needed to flip the decision boundary, or
                    np.nan if the decision boundary could not be flipped in 5 std from the value we got.
        """
        # Attempt to flip the decision by traversing the column in the opposite direction of the gradient
        deltas = []
        for j,col_idx in enumerate(q_col_idxs):
            qgrad_idx = self.qgrad_to_orig.index(col_idx)
            point_grad = qgrad[qgrad_idx]
            step = self.data_mat[:,col_idx].std()/100.0
            if step==0.0:
                print('Warning: The column you passed has no variance, and the step will be 1.0')
                step=1.0

            # Here we see the direction of the derivative and we want to go the other direction to flip the decision
            direc = -1*np.sign(point_grad)
            for i in range(1,max_itr):
                pred_data = np.copy(init_data)
                pred_data[:,col_idx] = init_data[:,col_idx]+direc*i*step
                new_pred = self.model.predict(pred_data.reshape(1,-1))

                if new_pred != init_pred:
                    delta = direc*i*step
                    deltas.append(delta)
                    break

                if (i+1)==max_itr:
                    print('Warning: decision flip not found in 5 std of point. col: {}'.format(q_col_names[j]))
                    deltas.append(np.nan)
        return deltas

    def _cat_follow_grad(self, cat_col_idxs, init_data, init_pred):
        """ Helper function to find the change needed in a categorical column to flip the decision
        boundary of the model in the class.
        -------
        Args: cat_col_idxs; iterable, contains the column indexes of the categorical columns to analyze
              init_data; numpy array, the initial row we are concerned with
              init_pred; int; (1 or 0 for binary classification) the original classification of our model
        -------
        Returns: list of bools; whether of not the categorical change flipped the decision boundary.
        """
        # Attempt to flip the decisions for categoricals (binary)
        changed_decisions = []
        for j,col_idx in enumerate(cat_col_idxs):

            pred_data = np.copy(init_data)
            pred_data[:,col_idx] = 0 if pred_data[:,col_idx][0]==1 else 1
            new_pred = self.model.predict(pred_data.reshape(1,-1))
            changed_decisions.append(all(new_pred!=init_pred))
        return changed_decisions

    def follow_gradient(self, point, col_names, df=True, max_itr=500, h=1e-4):
        """ Follow the opposite direction of the gradient to find the univariate movement
        of the specified feature that causes the decision to change. Note, this assumes any change
        in the initial decision counts as a decision change.

        NOTE: the maximum this function will search is five standard deviations in any direction.
                if the max_itr arg is set to 500
        --------
        Args: point; int, the row in our dataset we are interested in
              col_names; str or list; the string column name or a list of column names
              df; bool, is True returns a dataframe of the output and if False return arrays
              max_itr; int, the maximum number of iterations testing in any direction, this
                is the length it will search in any direction. In relation to the feature, set the
                value to 100x the number of std's you want to search from your initial point.
                (i.e. max_itr=500 is 5 std's from your point)
              h; float; the term to be added in finite difference.
        --------
        Returns: tuple of dataframes with quantitative features first and categorical features second
                    if df==True and if df==False  tuple of tuples with array of column names and array of
                    related change needed to flip the decision, for quantitative and categorical, respectively

        Note: If there are no categorical columns the function will still return a tuple but the categorical
                place will be None, and vice versa if there are no quantitative features passed.
        """
        qgrad, cgrad = self._grad(point, h=h)
        init_data = self.data_mat[point].reshape(1,-1)
        init_pred = self.model.predict(init_data)

        if isinstance(col_names, str):
            col_idxs = [self.cols.get_loc(col_names)]
            col_names = [col_names]
        else:
            col_idxs = [self.cols.get_loc(cn) for cn in col_names]

        # Here check for which passed columns are quantitative and which are cateogorical
        q_col_idxs, cat_col_idxs, q_col_names, cat_col_names = [],[],[],[]
        for ci,cn in zip(col_idxs,col_names):
            if ci in self.qgrad_to_orig:
                q_col_idxs.append(ci)
                q_col_names.append(cn)

            if ci in self.cgrad_to_orig:
                cat_col_idxs.append(ci)
                cat_col_names.append(cn)

        # Attempt to flip the decision by traversing the column in the opposite direction of the gradient
        deltas = self._quant_follow_grad(q_col_idxs, q_col_names, qgrad, init_data, init_pred, max_itr)

        # Attempt to flip the decisions for categoricals (binary)
        changed_decisions = self._cat_follow_grad(cat_col_idxs, init_data, init_pred)

        if df:
            qresults = pd.DataFrame({'Column_Names': q_col_names, 'Deltas': deltas})
            cat_results = pd.DataFrame({'Column_Names': cat_col_names, 'Decision_Changed': changed_decisions})
            if len(q_col_names) >0 and len(cat_col_names) >0:
                return qresults, cat_results
            if len(q_col_names)>0:
                return qresults, None
            if len(cat_col_names)>0:
                return None, cat_results
            else:
                raise ValueError('Proivide a column')
        return (q_col_names, deltas), (cat_col_names, changed_decisions)

    @classmethod
    def to_html(cls, quant_df, cat_df, flag='explain'):
        """ This function should take in a dataframe from the ig_explain and/or follow_gradients function
        and turn it into a fancy html output. """
        if flag not in {'explain', 'follow_grad'}:
            raise ValueError ('to_html does not accept flags except "follow_grad" or "explain"')

        raise NotImplementedError()

    def batch_fd(self,f,x,eps=1e-5):
        """ Helper: typical finite difference for n points and m features in the x matrix """
        n,nfeats = x.shape
        grad_1st = []

        for i in range(nfeats):
            if self.categorical[i]:
                continue
            # Compute f_plus[i]
            oldval = np.copy(x[:,i])
            x[:,i] = oldval + eps
            f_plus =  f(x)

            # Compute f_minus[i]
            x[:,i] = oldval - eps
            f_minus = f(x)

            # Restore
            x[:,i] = oldval

            grad_1st.append(np.ravel((f_plus - f_minus) / (2*eps)))
        return np.array(grad_1st)

    def batch_cat_fd(self,f,x):
        """Helper: finite difference for categorical values
        
        Inputs
        ------
        f: function to calculate fd for
        x: where we want to calculate the finite difference
        
        Returns
        -------
        grad_1st: numpy array with the computed finite differences.
        
        """
        n,nfeats = x.shape
        # Note: the below only works with binary
        flip = lambda col: np.abs(col-1)
        grad_1st = []

        for i in range(nfeats):
            if not self.categorical[i]:
                continue
            # Compute f_plus[i]
            oldval = np.copy(x[:,i])
            x[:,i] = flip(oldval)
            f_flip =  f(x)

            # Restore
            x[:,i] = oldval

            grad_1st.append(np.ravel(f_flip - f(x)))
        return np.array(grad_1st)

    def batch_grad(self, h=1e-4):
        """ Get the gradient of the model w.r.t. the specified point.
        -------
        Args: point; int, the row in our dataset we are interested in
              h; float; the term to be added in finite difference.
        -------
        Returns: tuple of numpy arrays; the gradient w.r.t the quantitative features and the
                    pseudo gradient w.r.t the categorical features of the model at the specified point
        -------
        NOTE: the input x must be a numpy ndarray.
        """
        data  = np.copy(self.data_mat)
        func = lambda x: self.model.predict_proba(x)[:,0]
        qgrad = self.batch_fd(func, data, eps=h)
        cgrad = self.batch_cat_fd(func, data)
        return qgrad, cgrad

    def batch_explain(self, include_cats=False, h=1e-4):
        """ This function gets the input gradients and finds the columns that are
        related to the number of specified max and min gradient values.  It then returns
        (max first) the columns associated with the maximum gradient and the values of
        the maximum gradient, and (min second) it returns the columns associated with
        the minimum gradient and the values of the minimum gradient.
        --------
        Args: point; int, the row in our dataset we are interested in
              num_top_preds; int, the number of top predictors we want to get back.
                i.e. a 2 here corresponds to the top two positive gradients and the top two
                negative gradients and their realted columns
              include_cats; bool; if True return a tuple of dataframes with quantitative and
                categorical features, respectively. if False return a dataframe of quant features.
              h; float, the term to be added in finite difference
        --------
        Returns: tuple of dataframes with the quantitative features first and the categorical
                    features second.
                 if include_cats==False it just returns the dataframe of quantitative features.
        --------
        Cite: Yotam Hechtlinger. 2016. Interpretation of prediction
                        models using the input gradient. arXiv preprint
                        arXiv:1611.07634
        """
        # Get input gradient
        qgrad, cgrad = self.batch_grad()

        qcol_names = [self.cols[self.qgrad_to_orig[i]] for i in range(len(qgrad))]
        qdf = pd.DataFrame(data=qgrad.T,columns=qcol_names)

        if not include_cats:
            return qdf

        ccol_names = [self.cols[self.cgrad_to_orig[i]] for i in range(len(cgrad))]

        if len(cgrad)>0:
            cdf = pd.DataFrame(data=cgrad.T,columns=ccol_names)
            return qdf,cdf
        return qdf,None

    def ig_explain(self, point, num_top_preds=1, include_cats=False, h=1e-4):
        """ This function gets the input gradients and finds the columns that are
        related to the number of specified max and min gradient values.  It then returns
        (max first) the columns associated with the maximum gradient and the values of
        the maximum gradient, and (min second) it returns the columns associated with
        the minimum gradient and the values of the minimum gradient.
        --------
        Args: point; int, the row in our dataset we are interested in
              num_top_preds; int, the number of top predictors we want to get back.
                i.e. a 2 here corresponds to the top two positive gradients and the top two
                negative gradients and their realted columns
              include_cats; bool; if True return a tuple of dataframes with quantitative and
                categorical features, respectively. if False return a dataframe of quant features.
              h; float, the term to be added in finite difference
        --------
        Returns: tuple of dataframes with the quantitative features first and the categorical
                    features second.
                 if include_cats==False it just returns the dataframe of quantitative features.
        --------
        Cite: Yotam Hechtlinger. 2016. Interpretation of prediction
                        models using the input gradient. arXiv preprint
                        arXiv:1611.07634
        """
        # Get input gradient
        qgrad, cgrad = self._grad(point, h=h)

        if len(qgrad) < 2.0*num_top_preds:
            raise ValueError('There are not enough quantitative predictors to satisfy your number of top predictors')

        # Get top positive and negative columnns for quantitative features
        qcol_maxs, qcol_mins = n_argmm(qgrad,size=num_top_preds)
        qcol_max_names = [self.cols[self.qgrad_to_orig[i]] for i in qcol_maxs]
        qcol_min_names = [self.cols[self.qgrad_to_orig[i]] for i in qcol_mins]

        # Get magnitude of the top positive and negative gradients for quantitative features
        qgrad_maxs, qgrad_mins = n_mm(qgrad, size=num_top_preds)
        qdf_dict = {'Positive_Columns': qcol_max_names, 'Positive_Gradients': qgrad_maxs, \
                        'Negative_Columns': qcol_min_names, 'Negative_Gradients': qgrad_mins }

        if not include_cats:
            return pd.DataFrame(qdf_dict)

        if include_cats and len(cgrad) < 2.0*num_top_preds:
            raise ValueError('There are not enough categorical predictors to satisfy your number of top predictors')

        # Get top positive and negative columns for categorical (binary) features
        cat_col_maxs, cat_col_mins = n_argmm(cgrad,size=num_top_preds)
        cat_col_max_names = [self.cols[self.cgrad_to_orig[i]] for i in cat_col_maxs]
        cat_col_min_names = [self.cols[self.cgrad_to_orig[i]] for i in cat_col_mins]

        # Get magnitude of the top positive and negative gradients for categorical features
        cat_grad_maxs, cat_grad_mins = n_mm(cgrad, size=num_top_preds)
        cat_df_dict = {'Positive_Columns': cat_col_max_names, 'Positive_Gradients': cat_grad_maxs, \
                        'Negative_Columns': cat_col_min_names, 'Negative_Gradients': cat_grad_mins }

        return pd.DataFrame(qdf_dict), pd.DataFrame(cat_df_dict)


def n_argmm(a,size,flag='both'):
    """ Find the n highest argmaxes / argmins of a 1D array.
    -------
    Args: a; 1D numpy array or 1D list
          size; int; the number of argmaxs / argmins to return
          flag; str, one of {'both','min','max'} specifies return type.
    -------
    Returns: if flag is 'min' returns argmins, if flag is 'max' returns argmaxs
               if flag is 'both' returns a tuple of the form (argmaxs, argmins)
    """
    a = np.array(a)

    if len(a.shape)!=1:
        raise ValueError('Only 1D input supported.')
    if flag not in set(['min','max','both']):
        raise ValueError('invalid flag {} should be one of "min","max","both"'.format(flag))
    srted = a.argsort()
    n_argmax = srted[-size:][::-1]
    n_argmin = srted[:size]

    if flag=='min':
        return n_argmin
    if flag=='max':
        return n_argmax
    return n_argmax, n_argmin

def n_mm(a,size,flag='both'):
    """ Find the n highest maxes / mins of a 1D array
    -------
    Args: a; 1D numpy array or 1D list
          size; int; the number of maxs / mins to return
          flag; str, one of {'both','min','max'} specifies return type.
    -------
    Returns: if flag is 'min' returns mins, if flag is 'max' returns maxs
               if flag is 'both' returns a tuple of the form (maxs, mins)
    """
    a = np.array(a)

    if len(a.shape)!=1:
        raise ValueError('Only 1D input supported.')
    if flag not in set(['min','max','both']):
        raise ValueError('invalid flag {} should be one of "min","max","both"'.format(flag))
    srted = np.sort(a)
    n_max = srted[-size:][::-1]
    n_min = srted[:size]

    if flag=='min':
        return n_min
    if flag=='max':
        return n_max
    return n_max, n_min

def plot_dice(df_topr, model, save_path=None, num_labels = 30, threshold = 0.5, figsize=(10,15)):
    """ This plots the DICE plot, which is a special histogram/cdf.
    What's special about it?  First, it's vertically oriented.  Secondly, the bars are colored based on if they're above
    or below the threshold.
    Most importantly, next to the histogram is text that shows the most likely positively- and negatively- correlated
    features at this range of the the loan-default probability distribution
    ------
    Args: df_x; pandas dataframe with the x data
          model; any valid model with a .predict method
          num_labels; number of "most important features" labels, evenly spaced along the pdf
          threshold: decision threshold for coloring histogram bars red
          figsize; 2-tuple, size of output plot
          num_subsamples; number of subsamples to select for building plot
    ------
    Returns: None
    """
    # Plot the top reasons df:
    fig, ax = plt.subplots(figsize=figsize)

    sns.kdeplot(df_topr.proba, color=sns.xkcd_rgb["green"], shade=True, vertical=True)
    n, bins, patches = plt.hist(df_topr.proba, density=True,
                                orientation='horizontal', color=sns.xkcd_rgb["green"],
                                alpha=0.3, bins=num_labels)

    # Set hist bars above threshold to red (surprisingly complicated...)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    threshold_norm = threshold / max(col)
    for b, c, p in zip(bins, col, patches):
        if b > threshold_norm:
            plt.setp(p, 'facecolor', 'darkred')

    # CDF in grey
    sns.distplot(df_topr.proba, color=sns.xkcd_rgb["grey"], vertical=True, hist=False,
                 kde_kws={'cumulative': True, 'alpha': 0.5, 'linestyle': '--'})

    # Chart visual params
    plt.xlim((0, 8))
    plt.ylim((-0.15, 1.15))
    plt.title("Most important cause of loan decisions")
    plt.xlabel("Probability of default")
    plt.grid(color='grey', linestyle=':', linewidth=1.5, alpha=0.2)

    # Next, put the labels above PDF
    label_placer = gaussian_kde(df_topr.proba.values)
    df_topr['quantile_bin'] = pd.qcut(df_topr.proba, num_labels, labels=False)

    for i in range(num_labels):
        # Determine whether the most frequent cause is a positive or negative variable, save it
        most_pos_bin = df_topr.loc[df_topr.quantile_bin == i, 'most_positive'].value_counts()
        most_neg_bin = df_topr.loc[df_topr.quantile_bin == i, 'most_negative'].value_counts()

        # Add most important feature labels to chart
        # TODO: FIND A WAY TO PLACE THE LABELS BETTER WITH MATPLOTLIB TRANSFORM()
        fig.text(label_placer(i / num_labels)/8 + 0.15, i/50 + 0.2, most_pos_bin.index[0], fontsize=12,
                 va='bottom', color="darkgreen")
        fig.text(label_placer(i / num_labels)/8 + 0.30, i/50 + 0.2, most_neg_bin.index[0], fontsize=12,
                 va='bottom', color="darkorange")

    # Titles for the labels
    fig.text(label_placer(1)/8+0.15, num_labels/50 + 0.2, '+ve feature:')#, fontsize=12, va='bottom', color="darkgreen")
    fig.text(label_placer(1)/8+0.30, num_labels/50 + 0.2, '-ve feature:')#, fontsize=12, va='bottom', color="darkorange")

    if save_path is not None:
        fig.savefig(save_path)

def plot_decision_boundary_2d(df_x, model, col1, col2, title, df_y=None,
                              samples=True, plot_proba=True, save_path=None, figsize=(8, 6)):
    """ This function plots the approximate decision boundary in the 2D space
    of the columns specified. It works by building a KD-Tree of the points made
    by the values of the specified columns and then looking up the closest point
    to the values we create on our interval for plotting. We use the feature values
    of the closest point as the feature values of the columns not being plotted
    when determining the decision boundary.  So the clarify this is an ESTIMATE.
    We also plot the actual points and decisions so the user can see where the estimate fails.
    ------
    Args: df_x; pandas dataframe with the x data
          model; any valid model with a .predict method
          col1; string representing the first column (x axis of the plot)
          col2; string representing the second column (y axis of the plot)
          title; string that is the title of the plot
          save_path; path to save the figure to. does not save if left on default which
                is None. Note: you have to include what you want to save it as with the extension
                e.g. save_path='../reports/decision_boundary.png'
          df_y; either numpy array or panads series with y values defaults to None
                and real points are not included
    ------
    Returns: None
    """
    fig, ax = plt.subplots(figsize=figsize)

    # DataFrame top numpy transition
    x = df_x.as_matrix()
    c1 = list(df_x.columns).index(col1)
    c2 = list(df_x.columns).index(col2)

    # Make the kd tree
    kdtree = BallTree(x[:, [c1, c2]], leaf_size=400)

    # Create mesh
    # Interval of points for feature 1
    min0 = x[:, c1].min()
    max0 = x[:, c1].max()
    interval0 = np.arange(min0, max0, (max0 - min0) / 100)
    n0 = np.size(interval0)

    # Interval of points for feature 2
    min1 = x[:, c2].min()
    max1 = x[:, c2].max()
    interval1 = np.arange(min1, max1, (max1 - min1) / 100)
    n1 = np.size(interval1)

    # Create mesh grid of points
    x1, x2 = np.meshgrid(interval0, interval1)
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    xx = np.concatenate((x1, x2), axis=1)

    idxs = kdtree.query(xx, k=1, return_distance=False)
    nearest_points = x[idxs.ravel()]
    nearest_points[:, c1] = x1.ravel()
    nearest_points[:, c2] = x2.ravel()

    if plot_proba is True:
        # Predict on mesh of points
        yy = model.predict_proba(nearest_points)[:, 0]
        yy = yy.reshape((n0, n1))

        # Plot decision surface
        x1 = x1.reshape(n0, n1)
        x2 = x2.reshape(n0, n1)
        levs = np.linspace(0, .8, 50)  # Bias the top because green is less distinct than red
        ax.contourf(x1, x2, yy, cmap='RdYlGn', alpha=1, levels=levs)

        # Plot scatter plot of data
        if df_y is not None and samples is True:
            y_true = np.array(df_y).reshape(-1, )
            ax.scatter(x[:, c1], x[:, c2], c='grey', label='Samples', zorder=1.1, alpha=0.05)
    else:
        # Predict on mesh of points
        yy = model.predict(nearest_points)
        yy = yy.reshape((n0, n1))

        # Plot hard decision
        x1 = x1.reshape(n0, n1)
        x2 = x2.reshape(n0, n1)
        ax.contourf(x1, x2, yy, cmap='RdGy_r', alpha=0.7)

        # Plot scatter plot of data
        if df_y is not None and samples is True:
            y_true = np.array(df_y).reshape(-1, )
            ax.scatter(x[y_true == 1, c1], x[y_true == 1, c2], c='darkgrey', label='Fully Paid', zorder=1, alpha=0.2)
            ax.scatter(x[y_true == 0, c1], x[y_true == 0, c2], c='darkred', label='Default', zorder=1.1, alpha=0.1)
            ax.legend()

            # Label axis, title
    ax.set_title(title)
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    if save_path is not None:
        fig.savefig(save_path)


################################################################
# LIME ERROR ANALYSIS
################################################################

def reconstruct_scaled_data(orig_data, scaled_data):
    """ Reconstruct the original data given both the original and the scaled data.
    The original data is used to get the scaling mean and std, and then the generated points
    from the LIME perturbation are transformed to be on the same scale as the original data.
    ---------
    Args: orig_data; numpy array, the dataset used in training of the model (i.e. the data
            passed to LimeTabularExplainer)
          scaled_data; numpy array; comes from the LIME explainer class and is the perturbed data on which
            the model is evaluated to determine influence of features on the final output.
    ---------
    Returns: numpy ndarray; the LIME perturbed data scaled back to the original scale.
    """
    train_mean = np.mean(orig_data, axis=0)
    train_std = np.std(orig_data, axis=0)
    npre = scaled_data * np.broadcast_to(train_std, shape=scaled_data.shape)
    return npre + np.broadcast_to(train_mean, shape=scaled_data.shape)

def finite_diff(f,x,eps=1e-5):
    """ Helper: typical finite difference for n points and m features in the x matrix """
    n,nfeats = x.shape
    grad_1st = [0]*nfeats

    for i in range(nfeats):
        # Compute f_plus[i]
        oldval = np.copy(x[:,i])
        x[:,i] = oldval + eps
        f_plus =  f(x)

        # Compute f_minus[i]
        x[:,i] = oldval - eps
        f_minus = f(x)

        # Restore
        x[:,i] = oldval

        grad_1st[i] = np.ravel((f_plus - f_minus) / (2*eps))
    return np.array(grad_1st)

def get_hess(model, x, eps=1e-5):
    """ Helper function for curvature """
    n,nfeats = x.shape
    jac = lambda x: finite_diff(model, x, eps=eps)
    h = finite_diff(jac, x, eps=eps)
    return h.reshape(nfeats,nfeats,n)

def get_hess_dets(hess):
    """ Helper function for Gaussian curvature """
    n = hess.shape[-1]
    dets = np.empty(n)
    for i in range(n):
        dets[i] = np.linalg.det(hess[:,:,i])
    return dets

def get_hess_traces(hess):
    """ Helper function for Mean curvature """
    n = hess.shape[-1]
    traces = np.empty(n)
    for i in range(n):
        traces[i] = hess[:,:,i].trace()
    return traces

def avg_gaussian_curvature(model,x,eps=1e-5):
    """ Gaussian curvature is the det of the hessian
    source http://www.math.union.edu/~jaureguj/principal_curvatures.pdf
    -------
    Args: model; function that when given a nxm input will return a nx1 output
          x; numpy array of size nxm where n is the number of datapoints and m is the number of features
          eps; the change to use in finite difference for the hessian
    -------
    Reutrns: float, the mean of the determinates of the hessians at each point (Gaussian curvature)
    """
    hess = get_hess(model,x,eps)
    dets = get_hess_dets(hess)
    return dets.mean()

def avg_mean_curvature(model,x,eps=1e-5):
    """ Mean curvature is the trace of the hessian
    source http://www.math.union.edu/~jaureguj/principal_curvatures.pdf
    -------
    Args: model; function that when given a nxm input will return a nx1 output
          x; numpy array of size nxm where n is the number of datapoints and m is the number of features
          eps; the change to use in finite difference for the hessian
    -------
    Reutrns: float, the mean of the traces of the hessians at each point (Gaussian curvature)
    """
    hess = get_hess(model,x,eps)
    traces = get_hess_traces(hess)
    return traces.mean()

def lime_local_curvature(model, orig_data, explain, scale=True, thresh=0.5, d_range=(0.55,0.45)):
    """ Calculate the local error of the LIME approximation for the point in the Explanation class.
        Error is calculated as 1-accuracy
    --------
    Args: model; sklearn model or any function with a valid '.predict_proba' method
          orig_data; numpy array, the dataset used in training of the model (i.e. the data
            passed to LimeTabularExplainer)
          explain; the instance of the Explainer class (in LIME) about the specific point of interest.
          scale; bool, if True scale the data back, if False leave the data passed to the
            original model without scaling it back to its original magnitude and variance.
          thresh; float; the threshold for classification, leave as default (0.5) if using
            standard sklearn (and most packages) defaults.
    --------
    Returns: float; the avergae Gaussian curvature of the real model
                over all the perturbed points
    """
    data = explain.scaled_data
    if scale:
        data = reconstruct_scaled_data(orig_data, data)

    # Get the prediction function w.r.t default prob from the real model
    mod_pred_prob = lambda x: model.predict_proba(x)[:,0]
    pred_vals = mod_pred_prob(data)
    msk = np.array([i and j for i,j in zip(d_range[0] > pred_vals, pred_vals > d_range[1])])

    #print(data[msk].shape)
    # Get the average Gaussian and Mean curvature for each perturbed point
    gauss_curvature = avg_gaussian_curvature(mod_pred_prob, data[msk], eps=1e-5)
    # Disregard mean curvature for now, but leave it in just in case we want it later
    # mean_curvature = get_hess_traces(hess).mean()
    return gauss_curvature

def lime_local_error_point(model, orig_data, explain, scale=True, thresh=0.5):
    """ Calculate the local error of the LIME approximation for the point in the Explanation class.
        Error is calculated as 1-accuracy
    --------
    Args: model; sklearn model or any function with a valid '.predict' method
          orig_data; numpy array, the dataset used in training of the model (i.e. the data
            passed to LimeTabularExplainer)
          explain; the instance of the Explainer class (in LIME) about the specific point of interest.
          scale; bool, if True scale the data back, if False leave the data passed to the
            original model without scaling it back to its original magnitude and variance.
          thresh; float; the threshold for classification, leave as default (0.5) if using
            standard sklearn (and most packages) defaults.
    --------
    Returns: float; the percent of equal predictions between the real model and the
                local linear approximation. i.e. the accuracy of the local linear
                approximation with the model prediction as the ground truth.
    --------
    NOTE: The explain input (a call of 'explain_instance' from the Explainer class) is inherently
            stochastic, even if you pass a random seed to the initial LimeTabularExplainer class.
    """
    data = explain.scaled_data
    if scale:
        data = reconstruct_scaled_data(orig_data, data)

    # Get the prediction from the real model
    mod_pred = model.predict(data)

    # The below is a bool vector (format due to LIME)
    lin_comb = np.sum([explain.scaled_data[:,i[0]]*i[1] for i in explain.local_exp[1]],axis=0) + explain.intercept[1]
    exp_pred = lin_comb > thresh

    if len(exp_pred)!= len(mod_pred):
        raise ValueError('The length of the predictions for the explanation and the model are not equal.')
    return np.sum(mod_pred!=exp_pred)/len(exp_pred)

def lime_local_error_dist(model, orig_data, explainer, pred_func, num_feat, scale=True, thresh=0.5):
    """ This function calculated the local linear approximation error for every point in the
    passed orig_data argument, and returns that value as a array of accuracy percentages.
    --------
    Args: model; sklearn model or any function with a valid '.predict' method
          orig_data; numpy array, the dataset used in training of the model (i.e. the data passed to LimeTabularExplainer)
          explainer; the instance of the LimeTabularExplainer class (in LIME) about the data.
          pred_func; function, the function to use in prediction for the model (it must give probabilities
            for each class)  i.e. in sklearn it would look like:
                        pred_func = lambda x: model.predict_proba(x).astype(float)
          num_feat; int, the number of features you want to have in the linear approximation.
            More features will likely have greater fidelity to the real model, but will be
            likely harder to intrepret.
          scale; bool, if True scale the data back, if False leave the data passed to the
            original model without scaling it back to its original magnitude and variance.
          thresh; float; the threshold for classification, leave as default (0.5) if using
            standard sklearn (and most packages) defaults.
    --------
    Returns: numpy array; the array of the percent of equal predictions between the
                real model and the local linear approximation. i.e. the accuracy of
                the local linear approximation with the model prediction as the
                ground truth at each point in orig_data.
    --------
    NOTE: The explain input (a call of 'explain_instance' from the Explainer class) is inherently
            stochastic, even if you pass a random seed to the initial LimeTabularExplainer class.
    """
    lime_errors = []
    for i in range(orig_data.shape[0]):
        expl = explainer.explain_instance(orig_data[i], pred_func, num_features=num_feat)
        lime_errors.append(lime_local_error_point(model, orig_data, expl, scale=scale, thresh=thresh))
    return np.array(lime_errors)

def lime_local_error_plot(dist, point_err):
    """ Plot the distribution of the local approx errors and a vertical line at the
    point specified.
    --------
    Args: dist; numpy array; the distribution of local approx errors
          point_err; float; the error at one specific point of interest in the distribution
    --------
    Returns: None; plots the distribution
    """
    sns.distplot(dist, kde=False)
    plt.axvline(point_err)
    plt.title('Distribution of local approximation errors in LIME.')
    sns.despine()
    plt.show()

def lime_local_error(model, orig_data, explainer, instance, pred_func, num_feat, scale=True, \
                     plot=True, thresh=0.5):
    """ Local error analysis for a single point w.r.t. the distribution of all local errors.
    --------
    Args: model; sklearn model or any function with a valid '.predict' method
          orig_data; numpy array, the dataset used in training of the model (i.e. the data passed to LimeTabularExplainer)
          explainer; the instance of the LimeTabularExplainer class (in LIME) about the data.
          instance; int; the index of the row of the specific instance in question
          pred_func; function, the function to use in prediction for the model (it must give probabilities
            for each class)  i.e. in sklearn it would look like:
                        pred_func = lambda x: model.predict_proba(x).astype(float)
          num_feat; int, the number of features you want to have in the linear approximation.
            More features will likely have greater fidelity to the real model, but will be
            likely harder to intrepret.
          scale; bool, if True scale the data back, if False leave the data passed to the
            original model without scaling it back to its original magnitude and variance.
          plot; bool, if True plot the distribution, if False do not plot
          thresh; float; the threshold for classification, leave as default (0.5) if using
            standard sklearn (and most packages) defaults.
    --------
    Returns: tuple; of the form (point_error, distribution_error) with the first argument
                as a float of the accuracy percentage between the local linear approximation
                and the original model. The second argument is a numpy array with the distribution
                of accuracies for every point in orig_data.
    --------
    NOTE: The explain input (a call of 'explain_instance' from the Explainer class) is inherently
            stochastic, even if you pass a random seed to the initial LimeTabularExplainer class.
    """
    # Get distribution of all local approx errors
    dist_er = lime_local_error_dist(model, orig_data, explainer, pred_func, num_feat, scale=scale, thresh=thresh)
    point_er = dist_er[instance]

    if plot:
        lime_local_error_plot(dist_er, point_er)

    return point_er, dist_er


class InterpManager():

    @classmethod
    def explain(cls, model, in_data, metrics=['LIME'], params={}, show_in_notebook=True):
        '''Function for single-model explanation, based on metrics in `metrics`. Defines a dictionary with keyword-function association.
        New interp functions can be added here.

        Inputs
        ------
        model : model object following the sklearn interface. Initialize with Initializer class for convenience.
        in_data : data used to train the model
        metrics : array of strings with metrics keywords. currently supported: LIME, IG. LICE
        params: dictionary of dictionaries with metric keywords and method parameters. e.g.: params = {'LIME':{'num_features':10, 'instances': [1,5]}}
        show_in_notebook : boolean, wether to print the explanation to notebook or not.

        Returns
        -------
        html_explanations: list of html objects containing a printable explanation produced by each metric requested in `metrics`.
                            There should be nxm explanations for n models and m metrics.

        '''

        # Add new functions here
        interp_functions = {'LIME': cls.explain_LIME,
                           'IG': cls.explain_IG,
                           'LICE': cls.explain_LICE}


        html_explanations = []

        for met in metrics:
            p={}
            if met in params: p = params[met]
            html_explanations.append(interp_functions[met](model, in_data, p, show_in_notebook))

        return html_explanations

    @classmethod
    def explain_all(cls, models, in_data, metrics=['LIME'], params ={}, show_in_notebook=True):
        '''General function for multi-model and multi-method explanation.

        Inputs
        ------
        models : list of model objects following the sklearn interface. Initialize them with Initializer class for convenience.
        in_data : data used to train the models
        metrics : array of strings with metrics keywords. currently supported: LIME, IG
        params: dictionary of dictionaries with metric keywords and method parameters. e.g.: params = {'LIME':{'num_features':10, 'instances': [1,5]}}
        show_in_notebook : boolean, wether to print the explanation to notebook or not.

        Returns
        -------
        html_explanations: list of html objects containing a printable explanation produced by each metric requested in `metrics`.
                            There should be nxm explanations for n models and m metrics.

        '''
        html_explanations = []

        for m in models:
            html_explanations.append(cls.explain(m, in_data, metrics, params, show_in_notebook))

        return html_explanations

    @staticmethod

    def explain_IG(model, in_data, ig_params={}, show_in_notebook=True):
        '''Method for input gradient explanation

        Inputs
        ------
        models : Model object following the sklearn interface. Initialize with Initializer class for convenience.
        in_data : Data used to train the model
        ig_params: Dictionary of parameters form lime computation. Ex: {'num_top_preds':1, 'instances': [1,5]}
        show_in_notebook : boolean, wether to print the explanation to notebook or not.

        Returns
        -------
        inputgrad.to_html(): HTML input gradient visualization

        '''
        
        # This function has not been fully implemented. We chose to use the InputGrad class instead 

        raise NotImplementedError()

        # These should be passed by the user as arguments (even if we choose stable defaults normally)
        num_top_preds=1
        include_cats=True
        h=1e-4
        inst = [0]
        model_dataframes = []

        if 'num_top_preds' in ig_params: num_top_preds = ig_params['num_top_preds']
        if 'include_cats' in ig_params: include_cats = ig_params['class_names']
        if 'h' in ig_params: h = ig_params['h']
        if 'instances' in ig_params: inst = ig_params['instances']
        if 'model_dataframes' in ig_params: model_dataframes = ig_params['model_dataframes']

        inputgrad = InputGrad(model, in_data)

        dataframes = []
        for i in inst:
            df_quant, df_cat = inputgrad.ig_explain(i, num_top_preds, include_cats, h)
            dataframes.append((df_quant, df_cat))

        model_dataframes.append(dataframes)

        return inputgrad.to_html()

    @classmethod
    def explain_LIME(cls, model, in_data, lime_params={}, show_in_notebook=True):
        '''Method for input gradient explanation

        Inputs
        ------
        models : Model object following the sklearn interface. Initialize with Initializer class for convenience.
        in_data : Data used to train the model
        ig_params: Dictionary of parameters form lime computation. Ex: {'num_top_preds':1, 'instances': [1,5]}
        show_in_notebook : boolean, wether to print the explanation to notebook or not.

        Returns
        -------
        inputgrad.to_html(): HTML input gradient visualization

        '''
        # default parameters
        n_feat = 5
        cl_names = ['Default','Paid']
        k_width = 3
        d_cont = True
        r_state = 135
        inst = [0]
        scale_p = True
        plot_p = True
        thresh_p = 0.5
        confidence_method = ''
        get_error_instances = False
        get_error_dataset = False
        error_dataset = []
        error_instances = []
        confidence = []

        if 'num_features' in lime_params: n_feat = lime_params['num_features']
        if 'class_names' in lime_params: cl_names = lime_params['class_names']
        if 'kernel_width' in lime_params: k_width = lime_params['kernel_width']
        if 'discretize_continuous' in lime_params: d_cont = lime_params['discretize_continuous']
        if 'random_state' in lime_params: r_state = lime_params['random_state']
        if 'instances' in lime_params: inst = lime_params['instances']
        if 'scale' in lime_params: scale_p = lime_params['scale']
        if 'plot' in lime_params: plot_p = lime_params['plot']
        if 'thresh' in lime_params: thresh_p = lime_params['thresh']
        if 'confidence_method' in lime_params: confidence_method = lime_params['confidence_method']
        if 'get_error_instances' in lime_params: get_error_instances = lime_params['get_error_instances']
        if 'get_error_dataset' in lime_params: get_error_dataset = lime_params['get_error_dataset']

        feat_names = in_data.columns
        in_data = in_data.as_matrix()

        lime_explainer = lime.lime_tabular.LimeTabularExplainer(in_data, \
                                                                feature_names=feat_names, \
                                                                class_names=cl_names, \
                                                                kernel_width=k_width, \
                                                                discretize_continuous=d_cont, \
                                                                random_state=r_state)

        pred_func = lambda x: model.predict_proba(x)
        exp_figures=[]
        exp_instances=[]
        for i in inst:
            exp_i = lime_explainer.explain_instance(in_data[i], pred_func, num_features=n_feat)
            if confidence_method:
                confidence.append(cls.confidence_LIME(model, in_data[i], exp_i, lime_explainer, n_feat = n_feat, method=confidence_method))
            if get_error_instances:
                error_instances.append(lime_local_error_point(model, in_data, exp_i, scale=True, thresh=0.5))
            if show_in_notebook:
                print('LIME explanation for model',model.__class__.__name__,'on instance',i)
                exp_i.show_in_notebook()

            exp_figures.append(exp_i.as_pyplot_figure())
            exp_instances.append(exp_i)

        if get_error_dataset:
            error_dataset = lime_local_error_dist(model, in_data, lime_explainer, pred_func, n_feat, scale=True, thresh=0.5)

        if get_error_instances:
            error_instances = lime_local_error_point

        return [exp_figures, exp_instances, confidence, error_instances, error_dataset]


    @classmethod
    def confidence_LIME(cls, model, observation, explanation, lime_explainer, n_feat=5, method='internal', n_perturb=50):
        '''Gateway method for computing confidence of the lime feature weights. Calls the method-specific confidence functions.'''

        if method == 'external':
            return cls.confidence_LIME_external(model, observation, explanation, lime_explainer, n_feat=n_feat, method=method, n_perturb=n_perturb)
        elif method == 'internal':
            return cls.confidence_LIME_internal(model, observation, explanation, lime_explainer, n_feat=n_feat, method=method, n_perturb=n_perturb)
        else:
            raise ValueError('Unknown confidence method provided.')

    @staticmethod
    def confidence_LIME_external(model, observation, explanation, lime_explainer, n_feat=5, method='internal', n_perturb=5):
        '''This method perturbs the observation, computes LIME for each perturbed point, and computes the top LIME feature of each perturbed point.\
        These feature weights are then used to compute a confidence interval for the LIME weights, which is currently calculated as the standard error (std dev) of the n_perturb computed weights.'''


        start = timer()
        perturbed_observations, inverse = lime_explainer._LimeTabularExplainer__data_inverse(observation, n_perturb)

        feature_weights={}

        for tup in explanation.local_exp[1]:
            k,v=tup
            feature_weights[k] = []


        for p in inverse:
            # We generate an explanation and get twice as many features as in the original explanation
            exp_i = lime_explainer.explain_instance(p, model.predict_proba, num_features=n_feat)  

            # We add the explanation weight to our dictionary of important feature weights
            for tup in exp_i.local_exp[1]:
                k,v = tup
                if k in feature_weights.keys():
                    feature_weights[k].append(v)
                    
#        After getting the lime weights of the {n_perturb} observations, we should have a feature_weight 
#        dictionary with {n_feat} keys (the top features of the original observation) and each key 
#        should map to an array of {n_perturb} observations - or less if some
#        perturbations didnt even contain the top 5 original influential features in their top 10 influential features!

        # We now use each vector of n_perturb weights to compute the standard error of each measure, as if it was a distribution. We compute the std:
        
        standard_errors,means={},{}
        for i,k in enumerate(feature_weights.keys()):
            standard_errors[k] = np.std(feature_weights[k])
            means[k] = np.mean(feature_weights[k])
            print('feature "',k,'" mean:',means[k],'std:',standard_errors[k], 'original:', explanation.local_exp[1][i])

        end = timer()
        print('\n\nTime spent in confidence_external:', end-start)
        return standard_errors


    def confidence_LIME_internal(model, observation, explanation, lime_explainer, n_feat=5, method='internal', n_perturb=5):
        '''This method computes LIME n_perturb times with different random seeds \
                       (which translates to different internal LIME perturbations) and uses the `n_perturb` feature\
                       weights computed in this fashion to calculate a confidence interval for the LIME feature \
                       weights of a given observation. '''

        start = timer()
        feature_weights={}

        for tup in explanation.local_exp[1]:
            k,v=tup
            feature_weights[k] = []


        c=0 ##DEBUG
        for i in range(n_perturb):
#            lime_explainer.random_state = check_random_state(np.random.randint(1e6))
            exp_i = lime_explainer.explain_instance(observation, model.predict_proba, num_features=n_feat)
            if c==0 or c==1:
#                print('\nPrinitng explanation.local_exp for explanation that was given to us (should be original observation)\n',explanation.local_exp)
#                print('\nPrinitng exp_i.local_exp for recomputation of LIME:',exp_i.local_exp)
                c+=1
            ###

            # We add the explanation weight to our dictionary of important feature weights
            for tup in exp_i.local_exp[1]:
                k,v = tup
                if k in feature_weights.keys():
                    feature_weights[k].append(v)


        standard_errors={}
        for i,k in enumerate(feature_weights.keys()):
            standard_errors[k] = np.std(feature_weights[k])
            print('feature "',k,'" mean:',np.mean(feature_weights[k]),'std:',standard_errors[k], 'original:', explanation.local_exp[1][i])

        end = timer()
        print('\n\nTime spent in confidence_internal:', end-start)
        return standard_errors






    @staticmethod
    def explain_LICE(model, in_data,  params={}, show_in_notebook=True):
        """Generates LICE figures andreturns them, given a model, full data and a set of params.

        Inputs
        ------
        model : model object following the sklearn interface. Initialize with Initializer class for convenience.
        in_data : data used to train the model.
        params: dictionary of LICE parameters. 2 params available: feature_names and observation. Feature_names corresponds
        to the features to plot LICE plots for. Observation is the number of the obs to analyze.
        show_in_notebook : boolean, wether to print the explanation to notebook or not.

        Returns
        -------
        figs: List of LICE plot figures, one per feature name.
        """

        feature_names = ['dti']
        observation = 0

        in_data = in_data.astype(np.float64)

        if 'feature_names' in params: feature_names = params['feature_names']
        if 'observation' in params: observation = params['observation']

        figs=[]
        for f in feature_names:
            figs.append( Lice.plot_lice(in_data, model, f, observation))

        return figs



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
        observation_x = df.iloc[observation,list(df.columns).index(col)]
        observation_i = np.abs(xbins - observation_x).argmin()
        observation_x_rounded = xbins[observation_i]

        # Predict observation at each feature value
        df_observation_whatif = df.iloc[[observation]*len(xbins),:].reset_index(drop=True)
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
                            color=cm.ocean(0.7 * (1 - int(y_col) / 100.)) if quantile_colorplot else '#1A4E5D',
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
