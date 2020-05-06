"""
This library contains a set of functions that help you detect similar 
images in your archive. It will detect the 'best' image per group using a 
high-pass filter and copy these to another folder.

Thus, you do not have to pre-select images for your foto show yourself 
(content is not considered a quality critera).
"""

import glob
import os
import shutil
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts
from sklearn.linear_model import LogisticRegression

version = '0.0.3'

def split_filepath(file, split=False):
    """
    Function to split file from path and return either of them.
    
    Parameters:
    ------------------------------
        file: (str), full path to file
        split: (bool), flag, if path shall be returned

    Returns:
    ------------------------------
        str, Filename
    """
    file = file.replace('\\', '/')
    if split is True:
        if os.path.isfile(file):
            _ = file.split('/')
            return '/'.join(_[:-2]), _[-1]
        else:
            if os.path.exists(file):
                if file[-1] is not '/':
                    file = file + '/'
                return file
            else:
                print('The file/path {} does not exist.'.format(file))
    else:
        return file
        
def read_files(path, ext):
    """
    Function to read image filenames from a directory and their creation date.
    
    Parameters:
    ------------------------------
        path: (str), filename
        ext: (str), extension for files to be considered

    Returns:
    ------------------------------
        list, Filenames
    """
    files = glob.glob('{}*.{}'.format(path, ext))
    # file modification date seems to be more reliable
    files = [(split_filepath(f, split=True)[-1], os.stat(f).st_mtime) for f in files]
    #files = [(split_filepath(f, split=True)[-1], os.path.getctime(f)) for f in files] 
    return files

def read_img(file, read_type=None):
    """
    Function to read images.
    
    Parameters:
    ------------------------------
        file: (str), filename
        read_type: (str), different opencv conversions of images
    
    Returns:
    ------------------------------
        array, Image
    """
    if read_type == None:
        return cv2.imread(file)
    elif read_type == 'hsv':
        return cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2HSV)
    elif read_type == 'gray':
        return cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)

def resize_img(img, scale=0.2):
    """
    Function resizes image.
    
    Parameters:
    ------------------------------
        img: (array), image
        scale: (float), factor used for reduction (1.0 means no change)
    
    Returns:
    ------------------------------
        array, Image
    """
    x, y = int(img.shape[0]*scale), int(img.shape[1]*scale)
    return cv2.resize(img, (y, x))

def calculate_hist(img, channels=[0], mask=None, histSize=[256], ranges=[0, 256]):
    """
    Function to calculate histograms for image data.
    
    Refer to cv2.calcHist of opencv library for more information.
    
    Returns:
    ------------------------------
        array, Histogramm of Image
    """
    return cv2.calcHist([img], channels=channels, mask=None, histSize=histSize, ranges=ranges)

def hash_image(img, min_pix=8):
    """
    Function that resizes images to icon size, leaving only the bare silhouette of the image.
    
    Parameters:
    ------------------------------
        img: (array), input image
        min_pix: (int, tuple), min value for the min axis
        
    Returns:
    ------------------------------
        array, Resized image.
    """
    if len(img.shape) == 2:
        y, x = img.shape
    elif len(img.shape) == 3:
        y, x, _ = img.shape
    
    if type(min_pix) == int:
        if x <= y:
            y = int(y/x * 8)
            x = 8
        else:
            x = int(x/y * 8)
            y = 8
            
    elif type(min_pix) == tuple:
        x, y = min_pix
        
    return cv2.resize(img, (x, y))


def copy_images(df, path=('./images/', './processed/'), rank_col=None, crit_col=None):
    """
    Function that copies the processed images to the destination repository.
    
    Parameters:
    ------------------------------
        df: (pandas DataFrame), that is grouped by column rank_col
        path = (tuple, str), (input_path, output_path) of images 
        rank_col: (str), ranked column in the DataFrame
        crit_col: (str), the column containing the criteria, greater
            values are better
    Returns:
    ------------------------------
        None      
    """
    for cur_file in df.loc[df.groupby([rank_col])[crit_col].idxmax(), 'file'].values:
        shutil.copy2(path[0] + cur_file, path[1] + cur_file)


def rotate_img(img, degree=0):
    """
    Function that rotates images.
    
    Parameters:
    ------------------------------
        img = (np.array), input image
        degree = (float), expected rotation degree (between 0° and 360°)
        
    Returns:
    ------------------------------
        array, Rotated image.
    """
    rows, cols = img.shape
    M_rot = cv2.getRotationMatrix2D((cols/2,rows/2), degree, 1)
    img_rot = cv2.warpAffine(img, M_rot, (cols,rows), 
                         flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    return img_rot

def warp_img(img, scale=0.0, how=None):
    """
    Function that warps images.
    
    Parameters:
    ------------------------------
        img = (np.array), input image
        scale = (float), needs to be in [0, 0.5), defines how much the image axis
                         is stretched
        how = (str), stretched axis, has to be in ("bottom", "top", "left", "right")
        
    Returns:
    ------------------------------
        array, Warped image.
    """
    rows, cols = img.shape
    
    if how == 'bottom':
        pinp = np.float32([[rows*scale,0],[rows*(1 - scale),0],[0,cols],[rows,cols]]) # squeeze bottom x
    elif how == 'top':
        pinp = np.float32([[0,0],[rows,0],[rows*scale,cols],[rows*(1 - scale),cols]]) # squeeze top x      
    elif how == 'left':
        pinp = np.float32([[0,cols*scale],[rows,0],[0,cols*(1-scale)],[rows,cols]]) # squeeze left side
    elif how == 'right':
        pinp = np.float32([[0,0],[rows,cols*scale],[0,cols],[rows,cols*(1-scale)]]) # squeeze right side
    else:
        print('Parameter how has to be in "bottom", "top", "left", "right".')

    pout = np.float32([[0,0], [rows,0], [0,cols], [rows,cols]])
    M_warp = cv2.getPerspectiveTransform(pinp, pout)
    img_warp = cv2.warpPerspective(img, M_warp, (cols,rows),
                              flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    return img_warp

def compare_hashes_adv(images, hash_dim=(8, 8), range_deg=(-5, 5, 1), 
                       warping=('left', 'right', 'top', 'bottom'), 
                       range_warp=(0.01, 0.06, 0.01), return_hash_only=False):
    """
    Advanced function that compares the hashes of consecutive images and takes
    small variations into account.
    
    Parameters:
    ------------------------------
        images = (list), list of images (arrays)
        hash_dim = (tuple), expected x & y pixel of the hashed images
        range_deg = (tuple), (min_value, max_value, step) for image rotation
        warping = (tuple), how to warp image, choose and combine ('left', 'right', 'top', 'bottom')
        range_warp = (tuple), (min_value, max_value, step) for image warping,
                        min/max_values have to be in [0, 0.5)
        return_hash_only = (bool), if True, only true image hash is calculated
        
    Returns:
    ------------------------------
        List of hash similarity of consecutive images.
    """
    def hash_checker(img_a, img_b):
        return sum([1 if i[0] == i[1] else 0 for i in zip(img_a, img_b)])
    
    if return_hash_only is True:
        images = [hash_image(img, min_pix=hash_dim) for img in images]

        img_mean = [np.mean(img) for img in images]
        imgs_reshaped = [img.reshape(-1) for img in images]

        hashes = []
        for enum, cur_img in enumerate(imgs_reshaped):
            hashes.append([1 if px > img_mean[enum] else 0 for px in cur_img])

        compared_hashes = [hash_checker(*pair) for pair in zip(hashes, hashes[1:] + hashes[:1])]
    
    else:
        images_adv = []
        for img in images:
            images_ = [img]
            images_ += [rotate_img(img, deg) for deg in range(*range_deg)]
            images_ += [warp_img(img, scale, how=how) for how in warping 
                                                        for scale in np.arange(*range_warp)]
            images_adv.append([hash_image(img, min_pix=hash_dim).reshape(-1) for img in images_])
        
        img_mean = [np.mean(img) for images in images_adv for img in images]

        ix, iy = len(images), len(images_adv[0])
        img_mean = np.array(img_mean).reshape(ix, iy)

        hashes = []
        for enum, cur_img in enumerate(images_adv):
            var = []
            for enum2, variant in enumerate(cur_img):
                var.append([1 if px > img_mean[enum][enum2] else 0 for px in variant])
            hashes.append(var)

        def hash_checker(img_a, img_b):
            return sum([1 if i[0] == i[1] else 0 for i in zip(img_a, img_b)])

        compared_hashes = []
        for pair in zip(hashes, hashes[1:] + hashes[:1]):
            a, b = pair
            max_hash = 0
            for img_b in b:
                cur_hash = hash_checker(a[0], img_b)
                if cur_hash > max_hash:
                    max_hash = cur_hash
            compared_hashes.append(max_hash)
    return compared_hashes

def high_pass_filtering(img, x_shift=30, y_shift=30):
    """
    High-pass filter for images, calculates the magnitude spectrum.
    
    Parameters:
    ------------------------------
        img: (array), two dimensional image data
        x_shift, y_shift: (int), filter threshold, 0 means no filtering at all
        
    Returns:
    ------------------------------
        array, Magnitude spectrum of the image
    """
    rows, cols = img.shape
    _row, _col = int(rows/2), int(cols/2)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift[_row - y_shift:_row + y_shift, _col - x_shift:_col + x_shift] = 0 # here happens the filtering
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    flattened_spectrum = img_back.reshape(-1, 1)
    x_hist, y_hist = np.histogram(flattened_spectrum, bins=255)
    y_hist_corr = [(y_hist[i] + y_hist[i+1])/2.0 for i in range(len(y_hist)-1)]
    mag_spectrum = np.cumsum(y_hist_corr)[-1]
    return mag_spectrum

def calculate_performance(labels, predictions, n_img=None):
    """
    Returns performance summary for ranking evaluation - can be used for calibration.
    Has two contributions:
    
    Parameters:
    ------------------------------
        labels = (series, array), true labels
        predictions = (series, array), predicted labels
        n_img = (int), true number of groups
        
    Returns:
    ------------------------------
        tuple, (image groups found, 
                  unique image groups found, 
                  true number of image groups
                  normalized reduction ratio (1 is best ))
    """
    df_perf = pd.DataFrame(np.c_[labels, predictions], columns=['labels', 'pred'])
    df_grouped = df_perf.groupby(['pred'])['labels'].apply(lambda x: len(set(x))).to_frame()

    groups_found = min(n_img, len(set(df_grouped.index[df_grouped['labels']==1].values)))
    groups_unique = len(df_perf['pred'].unique())
    
    max_red = n_img / len(df_perf)
    reduction = groups_unique / len(df_perf) * groups_found / n_img
    reduction /= max_red
    
    return (groups_found, 
            groups_unique,
            n_img,
            reduction)
            
def calc_correlations(images, method):
    """
    Function to calculate correlations between images.
    
    Parameters:
    ------------------------------
        images: list, of images read in HSV mode
        method: str, 'bhattacharyya', 'correl' refer to cv2 for further information

    Returns:
    ------------------------------
        List of calculated correlation measures.
    """
    methods = {'bhattacharyya': cv2.HISTCMP_BHATTACHARYYA,
               'correl': cv2.HISTCMP_CORREL,
    }

    img_hists = [calculate_hist(cur_img, [0,1], None, [180, 256], [0,180, 0,256]) for cur_img in images]
    cors = [cv2.compareHist(img_hists[i-1], img_hists[i], methods[method]) for i in range(len(img_hists))]
    cors.append(cors.pop(0))

    if method == 'bhattacharyya':
        cors = list(map(lambda x: 1 - x, cors))
    return cors


def timelag_ranker(series, max_lag=5, max_per_group=5):
    def timelag_ranker_(series, max_lag, max_per_group):
        """
        Ranker function, distinguishes images based on difference timestamps.
        
        Parameters:
        ------------------------------
            series = pandas Series of timestamps
            max_lag = (float), seconds between images
            group_default = (int), maximum number of images per group
        
        Returns:
        ------------------------------        
            Generator of Rank
        """
        rank = 1
        n_group = 0
        for row in series.iloc[:-1]:
            yield rank
            n_group += 1
            if row > max_lag:
                rank += 1
                n_group = 0
            elif n_group > max_per_group:
                rank += 1
                n_group = 0
        yield rank
    return list(timelag_ranker_(series, max_lag, max_per_group))

def hash_ranker(series, dim=None, limit=0.875):
    def hash_ranker_(series, dim, limit):
        """
        Ranker function, distinguishes images based on similary of image hashes.
        Hash refers here to low resolution images (~8 px per x and y dimensions)
        
        Parameters:
        ------------------------------
            series = pandas Series, containing image hash values
            dims = (int), number of image dimensions (in px)
            limit = (float), lower limit for hash similarity to be recognized 
                            as similar imageS
        
        Returns:
        ------------------------------        
            Generator of Rank
        """
        rank = 1
        for row in series.iloc[:-1]:
            yield rank
            if row < np.product(dim) * limit:
                rank += 1
        yield rank
    return list(hash_ranker_(series, dim, limit))

def corr_ranker(series, limits={'bhattacharyya': 0.61, 'correl': 0.8}):
    def corr_ranker(series, limits):
        """
        Ranker function, distinguishes images based on correlation of image (has to be provided).
        
        Parameters:
        ------------------------------
            series = pandas Series, containing image hash values
            limits = (dict), where keys refer to the method used and the values
                                are the lower bounds for recognition as similar images.
        
        Returns:
        ------------------------------        
            Generator of Rank
        """
        name = series.name.split('_')[0]
        rank = 1
        for row in series.iloc[:-1]:
            yield rank
            if row < limits[name]:
                rank += 1
        yield rank     
    return list(corr_ranker(series, limits))

def vote_ranker(series, limit=0.41):
    def vote_ranker_(series, limit):
        """
        Ranker function, mean of differences of ranks (from different methods)
        serves as discriminator of ranks from ensemble.
        
        Parameters:
        ------------------------------
            series = pandas Series, containing image average 
            limit = (dict), has to be in [0, 1]
        
        Returns:
        ------------------------------        
            Generator of Rank
        """
        rank = 1
        for row in series.iloc[:-1]:
            yield rank
            if row < limit:
                rank += 1
        yield rank
    return list(vote_ranker_(series, limit))

def calculate_axis_ratio(img):
    """
    Function returns axis-ratio of image.

    Parameters:
    ------------------------------
        img: (array), input image
    
    Returns:
    ------------------------------        
       float, Axis ratio (horizontal/vertical)
    """
    if len(img.shape) == 2:
        h, v = img.shape
    elif len(img.shape) == 3:
        h, v, _ = img.shape
    else:
        h, v = img.shape[0], img.shape[1]
        
    return h/v

def img_shape_ranker(series, limit=0.01):
    def img_shape_ranker_(series, limit):
        """
        Ranker function, checks if a the image axis ratio changes (horizontal to vertical
        and vice versa).
        
        Parameters:
        ------------------------------
            series = pandas Series, containing image average 
            limit = (dict), has to be in (0, np.inf]
        
        Returns:
        ------------------------------        
            Generator of Rank
        """
        rank = 1
        for row in series.iloc[:-1]:
            yield rank
            if abs(row) >= limit:
                rank += 1
        yield rank
        
    return list(img_shape_ranker_(series, limit))


def batch_hashing(df, n_dims=(8, 64)):
    """
    Function that calculates the image hashes for different image dimensions.
    
    Parameters:
    ------------------------------
        df: (pandas DataFrame), dataframe containing at least the following columns:
            target, gray_images
            
        n_dims: (tuple, int), (min_value, max_value) for quadratic image hashes

    Returns:
    ------------------------------
        list of pandas DataFrames    
    """
    df = df.copy()
    targets = np.arange(0, len(df['target'].unique()))

    runs = []
    for i in range(*n_dims, 1):
        df['hash_value'] = compare_hashes_adv(df['gray_images'], hash_dim=(i, i),
                                                 return_hash_only=True)
        df['hash_value'] /= (i*i) 
        runs.append(df[['target', 'hash_value']].copy())
        
    return runs

def return_hashing_dist(data, _type=None, target_col=None, comp_col=None):
    """
    Function that calculates the hash values for (first) similar & non-similar images
    
    Parameters:
    ------------------------------
        data: (list of pandas DataFrame), each dataframe is expected to contain
                at least the following columns:
                target, hash_value
            
        _type: (str), choose between 'similar' and 'nonsimilar' 
        
        target_col: (str), column name of the target column
        
        comp_col: (str), column name of the column used for comparison

    Returns:
    ------------------------------
        list of pandas DataFrames
    """
    if _type == 'similar':
        similar = []
        for cur_res in data:
            #TODO: calculate mean of element [0,n-1] if group size > 2
            rel_rows = (cur_res.groupby([target_col])[comp_col].agg('count') > 1).values
            similar.append(cur_res.groupby([target_col])[comp_col].first().iloc[rel_rows])
        return similar
    
    elif _type == 'nonsimilar':
        nonsimilar = np.array([cur_res.groupby([target_col])[comp_col].last().values 
                  for cur_res in data])
        return nonsimilar

def make_logreg_fit(similar, nonsimilar, make_plot=True, 
               labels=('x', 'Probability'), limits=None):
    """
    Function that performs a logistic regression fit on the groups of similar
    and non-similar images and provides the threshold value (+ plot).
        
    Parameters:
    ------------------------------
        similar: (array), containing the hash values found
                            for similar groups
        nonsimilar: (array), containing the hash values found
                            for non-similar groups
        make_plot: (bool), create plot if True
        labels: (tuple), x- and y-label for the plot
        limits: (tuple), (min_x, max_x) for plots if not None
                            
        
    Returns:
    ------------------------------
        float, threshold.
    """
    X_train = np.append(nonsimilar, similar)
    X_train = np.c_[X_train, np.ones(len(X_train))]
    y_train = np.array(len(nonsimilar)*[0] + len(similar)*[1])
    
    if limits is None:
        min_val, max_val = min(X_train[:,0]), max(X_train[:,0])
    else:
        min_val, max_val = limits

    lreg = LogisticRegression()
    lreg.fit(X_train, y_train)

    x_vals = np.arange(min_val*0.8, max_val*1.2, 0.001)
    probs = lreg.predict_proba(np.c_[x_vals, np.ones(len(x_vals))])
    lower_bound = np.argmax(probs[probs<=0.50])
    
    if make_plot is True:
        fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                               gridspec_kw={'height_ratios': [5, 1]},
                               tight_layout=True)
        
        for paxis in range(2):
            axs[paxis].plot([x_vals[lower_bound], x_vals[lower_bound]], [-0.1, 1.1], '-',
                 color='gray', alpha=0.5, lw=10)
        
        axs[0].plot(x_vals, probs[:,1], 'b-', label='Probability curve', alpha=0.5)
        
        axs[0].scatter(X_train[y_train==0, 0], lreg.predict_proba(X_train)[y_train==0, 1],
                    marker='.', s=400, ec='gray', label='False', color='red', alpha=.8)
        axs[0].scatter(X_train[y_train==1, 0], lreg.predict_proba(X_train)[y_train==1, 1],
                    marker='.', s=400, ec='gray', label='True', color='green', alpha=.8)
        
        axs[1].eventplot(similar, lineoffsets=[0.2], linelengths=0.4, linewidths=0.5,
                         orientation='horizontal', color='green', alpha=0.5)
        axs[1].eventplot(nonsimilar, lineoffsets=[0.75], linelengths=0.4, linewidths=0.5,
                         orientation='horizontal', color='red', alpha=0.5)

        axs[0].axis([min_val*0.8, max_val*1.1, -0.1, 1.1])
        axs[0].grid()
        axs[0].legend(loc='upper left')

        xl, yl = labels
        axs[1].set_xlabel(xl)
        axs[0].set_ylabel(yl)
        axs[0].set_title('Logistic regression fit')
        
        axs[1].axis([min_val*0.9, max_val*1.1, -0.1, 1.1])
        axs[1].set_ylabel('Event')
        axs[1].tick_params(labelleft=False)
        axs[1].grid()
        plt.show()
    
    threshold = x_vals[lower_bound]
    print('Limit at {:.2f}'.format(threshold))
    return threshold


def bootstrap_data(df, n_runs=10):
    """
    This function shuffles the input data and repeatedly calculates the
    ranks using a variety of methods. This provides better estimates on
    the average values for both image groups (similar or non-similar).
    
    Parameters:
    ------------------------------
        df: (pandas DataFrame), dataframe containing at least the following columns:
            target, creation_date, hash_value, correl_corr, bhattacharyya_corr
            
        n_runs: (int), how often the experiment is repeated

    Returns:
    ------------------------------
        list of pandas DataFrames
    """
    true_targets = df['target'].copy()
    targets = np.arange(0, len(true_targets.unique()))

    runs = []
    for i in range(n_runs):
        np.random.seed(i)
        new_targets = np.random.choice(targets, size=len(targets), replace=False)
        
        # add some "randomness" by reversing images of each group
        if i % 2 == 0:
            df.sort_values(['creation_date'], inplace=True)
        else:
            df.sort_values(['creation_date'], ascending=False, inplace=True)
            
        df['target'] = true_targets.map(dict(zip(targets, new_targets)))
        df.sort_values(['target'], inplace=True)
        
        df['hash_value'] = compare_hashes_adv(df['gray_images'].tolist(),
                                                 return_hash_only=True)
        df['correl_corr'] = calc_correlations(df['hsv_images'].tolist(),
                                                 'correl')
        df['bhattacharyya_corr'] = calc_correlations(df['hsv_images'].tolist(), 
                                                'bhattacharyya')
        
        runs.append(df[['target', 'creation_date', 
                        'hash_value', 'correl_corr', 'bhattacharyya_corr']])
    return runs

def return_dist(data, _type=None, target_col=None, comp_col=None):
    """
    Function that calculates the comparison values for (first) similar & non-similar images
    
    Parameters:
    ------------------------------
        data: (list of pandas DataFrame), each dataframe is expected to contain
                at least the following columns:
                target, hash_value
            
        _type: (str), choose between 'similar' and 'nonsimilar' 
        
        target_col: (str), column name of the target column
        
        comp_col: (str), column name of the column used for comparison

    Returns:
    ------------------------------
        list of pandas DataFrames
    """
    if _type == 'similar':
        similar = []
        for cur_res in data:
            #TODO: calculate mean of element [0,n-1] if group size > 2
            rel_rows = (cur_res.groupby([target_col])[comp_col].agg('count') > 1).values
            similar.append(cur_res.groupby([target_col])[comp_col].first().iloc[rel_rows])
        similar = np.sort(np.array(similar).reshape(-1))
        return similar
    
    elif _type == 'nonsimilar':
        nonsimilar = np.sort(np.array([cur_res.groupby([target_col])[comp_col].last().values 
                  for cur_res in data]).reshape(-1))
        return nonsimilar

def plot_distributions(similar, nonsimilar, bins=10, labels=('x', 'y'), title=''):
    """
    Function that plots the histogram + kernel density functions 
    of comparison value for (first) similar & non-similar distributions.
    
    Parameters:
    ------------------------------
        similar: (array), containing the hash values found
                            for similar groups
        nonsimilar: (array), containing the hash values found
                            for non-similar groups
        bins: (int), number of bins in histogram
        label: (tuple), x- and y-label for plot
        title: (str), plot title

    Returns:
    ------------------------------
        None
    """
    full_vals = np.append(similar, nonsimilar)
    xmin, xmax = min(full_vals), max(full_vals)
    margin = (xmax - xmin)*0.1
    xrange = np.arange(xmin - margin*2, xmax + margin*2, 0.01)
        
    plt.figure(figsize=(15, 5))
    kde_sim = sts.gaussian_kde(similar)
    plt.hist(similar, bins=bins, rwidth=0.9, density=True, 
             label='first', color='gold', alpha=0.7);
    plt.plot(xrange, kde_sim(xrange), lw=2, ls='-', 
             color='#6666ff', label='similar-KDE')

    kde_nonsim = sts.gaussian_kde(nonsimilar)
    plt.hist(nonsimilar, bins=bins, rwidth=0.9, density=True, 
             label='last', color='gray', alpha=0.7);
    plt.plot(xrange, kde_nonsim(xrange), lw=2, ls='-', 
             color='#ff6666', label='last-KDE')
    
    plt.xlim([xmin - margin*2, xmax + margin*2])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def performance_report(similar, nonsimilar, limit=None, add_std=False):
    """
    Prints a performance report.
    
    Parameters:
    ------------------------------
    
        similar: (array), containing the hash values found
                            for similar groups
        nonsimilar: (array), containing the hash values found
                            for non-similar groups
        limit: (float), threshold value, if None, mean is used
        add_std: (bool), if True, standard deviation is added to the
                         mean threshold, works only if limit is None
                         
    Returns:
    ------------------------------
        None.
    
    """
    def precision_score(tp, fp, eps=1e-10):
        return (tp + eps) / (tp + fp + eps) 
    
    def recall_score(tp, fn, eps=1e-10):
        return (tp + eps) / (tp + fn + eps)
    
    def f1_score(tp, fp, fn, eps=1e-10):
        return 2 / (1/precision_score(tp, fp) + 1/recall_score(tp, fn))
    
    if limit is None:
        limit = similar.mean()
        if add_std is True:
            limit += similar.std()
        
    tp = sum([1 for c in similar if c >= limit])
    fn = len(similar) - tp
    tn = sum([1 for c in nonsimilar if c < limit])
    fp = len(nonsimilar) - tn

    print('Performance report\n'+'-'*50)
    print('True positive: {} -- False negative: {}'.format(tp, fn))
    print('True negative: {} -- False positive: {}'.format(tn, fp))
    print('\nPrecision score: {:.4f}'.format(precision_score(tp, fp)))
    print('Recall score: {:.4f}'.format(precision_score(tp, fn)))
    print('F1 score: {:.4f}'.format(f1_score(tp, fp, fn)))