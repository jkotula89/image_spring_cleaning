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

version = '0.0.2'

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
        Resized image.
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
        Rotated image.
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
        Warped image.
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
        Magnitude spectrum of the image
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

def calculate_loss(labels, predictions, n_img=None):
    """
    Loss function for ranking evaluation - can be used for calibration.
    Has two contributions:
    * MSE estimate, if the ranking yields more groups than actually exist (unproblematic)
    * a polynomial penalty, if the ranking groups together images of different classes (problematic,
        since images might get lost)
    
    Parameters:
    ------------------------------
        labels = (series, array), true labels
        predictions = (series, array), predicted labels
        n_img = (int), true number of groups
        
    Returns:
    ------------------------------
        tuple, (image groups found, 
                  unique image groups found, 
                  loss value (min value = 1),
                  reduction ratio (smaller is better))
    """
    def penalty(x):
        return np.sum( x**10 )
    
    df_perf = pd.DataFrame(np.c_[labels, predictions], columns=['labels', 'pred'])

    # 1. MSE loss for finding more image groups than are actually present
    mse = np.sum( (df_perf['labels'] - df_perf['pred']) ** 2)
    
    # 2. Penalty loss for missed images
    df_grouped = df_perf.groupby(['pred'])['labels'].apply(lambda x: len(set(x))).to_frame()
    miss_loss = penalty(df_grouped['labels'].values)
    
    # 3. Number of groups found
    groups_found = min(n_img, len(set(df_grouped.index[df_grouped['labels']==1].values)))
    groups_unique = len(df_perf['pred'].unique())
    groups_true = len(df_perf['labels'].unique())
    
    # 4. Reduction ratio
    if groups_found >= groups_true:
        reduction = groups_unique / len(df_perf)
    else:
        reduction = groups_found / len(df_perf)
    
    mse /= max(1, groups_found) * 100
    miss_loss /= max(1, groups_found) 

    return (groups_found, groups_unique, 
            mse + miss_loss, reduction)
def calc_correlations(images, method):
    """
    Function to calculate correlations between images.
    
    Parameters:
    ------------------------------
        images: list, of images read in HSV mode
        method: str, 'bhattacharyya' or 'correl', refer to cv2 for further information

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
        rank = 0
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
        rank = 0
        for row in series.iloc[:-1]:
            yield rank
            if row < np.product(dim) * limit:
                rank += 1
        yield rank
    return list(hash_ranker_(series, dim, limit))

def corr_ranker(series, coff_lim={'bhattacharyya': 0.61, 'correl': 0.8}):
    def corr_ranker(series, coff_lim):
        """
        Ranker function, distinguishes images based on correlation of image (has to be provided).
        
        Parameters:
        ------------------------------
            series = pandas Series, containing image hash values
            coff_lim = (dict), where keys refer to the method used and the values
                                are the lower bounds for recognition as similar images.
        
        Returns:
        ------------------------------        
            Generator of Rank
        """
        name = series.name.split('_')[0]
        rank = 0
        for row in series.iloc[:-1]:
            yield rank
            if row < coff_lim[name]:
                rank += 1
        yield rank     
    return list(corr_ranker(series, coff_lim))

def vote_ranker(series, lim=0.4):
    def vote_ranker_(series, lim):
        """
        Ranker function, mean of differences of ranks (from different methods)
        serves as discriminator of ranks from ensemble.
        
        Parameters:
        ------------------------------
            series = pandas Series, containing image average 
            lim = (dict), has to be in [0, 1]
        
        Returns:
        ------------------------------        
            Generator of Rank
        """
        rank = 0
        for row in series.iloc[:-1]:
            yield rank
            if row < lim:
                rank += 1
        yield rank
    return list(vote_ranker_(series, lim))