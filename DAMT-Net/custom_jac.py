import numpy as np
from skimage.util import img_as_ubyte
from skimage import io
from glob import glob
import shutil

def create_dir(dir):
    '''
    Create a directory if it does not exist

    Args:
      dir: The directory that will be created
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)

def rm_dir(dir):
    '''
    Remove a directory

    Args:
      dir: the directory to be removed
    '''
    if os.path.exists(dir):
        shutil.rmtree(dir)

def get_xy_image_list(dir):
    '''
    Reads the training images and labels from the specified directory.
    Where 'dir'/x/ contains all images, and 'dir'/y/ contains all labels.
    Image and labels will be sorted by filename

    Args:
      dir (str): The directory where the images are stored

    Returns:
      two lists, one containing the training images and the other containing the corresponding labels.
    '''

    if dir[-1]=='/':
        dir = dir[:-1]
    # Paths to the training images and their corresponding labels
    train_input_path = dir + '/x/*.*'
    train_label_path = dir + '/y/*.*'

    # Read the list of file names
    train_input_filenames = glob(train_input_path)
    train_input_filenames.sort()

    train_label_filenames = glob(train_label_path)
    train_label_filenames.sort()

    print( 'Input images loaded: ' + str( len(train_input_filenames)) )
    print( 'Label images loaded: ' + str( len(train_label_filenames)) )

    # read training images and labels
    train_img = [ img_as_ubyte( io.imread( x, as_gray=True ) ) for x in train_input_filenames ]
    train_lbl = [ img_as_ubyte( io.imread( x, as_gray=True ) ) for x in train_label_filenames ]

    return train_img, train_lbl

def get_image_list(dir):
    '''
    Reads all the images in the specified directory and returns a list of numpy arrays representing the
    images

    Args:
      dir: The directory that contains the images.

    Returns:
      A list of numpy arrays representing the images.
    '''

    train_label_path = dir + '/*.*'

    train_label_filenames = glob(train_label_path)
    train_label_filenames.sort()

    print( 'Label images loaded: ' + str( len(train_label_filenames)) )

    # read training images and labels
    train_lbl = [ img_as_ubyte( io.imread( x, as_gray=True ) ) for x in train_label_filenames ]
    return train_lbl

def jaccard_index_numpy(y_true, y_pred):
    """Define Jaccard index.
       Parameters
       ----------
       y_true : N dim Numpy array
           Ground truth masks. E.g. ``(num_of_images, x, y, channels)`` for 2D images or
           ``(volume_number, z, x, y, channels)`` for 3D volumes.
       y_pred : N dim Numpy array
           Predicted masks. E.g. ``(num_of_images, x, y, channels)`` for 2D images or
           ``(volume_number, z, x, y, channels)`` for 3D volumes.
       Returns
       -------
       jac : float
           Jaccard index value.
    """
    y_pred_ = (y_pred > 0.5).astype(np.uint8)

    TP = np.count_nonzero(y_pred_ * y_true)
    FP = np.count_nonzero(y_pred_ * (y_true - 1))
    FN = np.count_nonzero((y_pred_ - 1) * y_true)

    if (TP + FP + FN) == 0:
        jac = 0
    else:
        jac = TP / (TP + FP + FN)

    return jac

from PIL import Image

def add_padding(np_img):
    '''
    Given a numpy array, add padding to the image so that the image is a multiple of 256x256

    Args:
      np_img: the image to be padded

    Returns:
      A numpy array of the image with the padding added.
    '''
    image = Image.fromarray(np_img)
    height, width = np_img.shape

    if not width%256 and not height%256:
        return np_img

    x = width/256
    y = height/256

    new_width = int(np.ceil(x))*256
    new_height = int(np.ceil(y))*256

    left = int( (new_width - width)/2 )
    top = int( (new_height - height)/2 )

    result = Image.new(image.mode, (new_width, new_height), 0)

    result.paste(image, (left, top))

    return np.asarray(result)

def remove_padding(np_img, out_shape):
    '''
    Given an image and the shape of the original image, remove the padding from the image

    Args:
      np_img: the image to remove padding from
      out_shape (int,int): the desired shape of the output image (height, width)

    Returns:
      The image with the padding removed.
    '''
    height, width = out_shape # original dimensions
    pad_height, pad_width = np_img.shape # dimensions with padding

    if not width%256 and not height%256: # no hacia falta padding --> no tiene
        return np_img

    rm_left = int( (pad_width - width)/2 )
    rm_top = int( (pad_height - height)/2 )

    rm_right = pad_width - width - rm_left
    rm_bot = pad_height - height - rm_top

    return np.array(np_img[rm_top:-rm_bot, rm_left:-rm_right])

import os
def custom_test(gt_path, pred_path):
    '''
    It calculates the mean IoU for the test set.

    Args:
      gt_path: Path to ground truth images
      pred_path: path to the folder containing the predicted images

    Returns:
      The mean IoU of the test set.
    '''
    test_lbl = get_image_list(os.path.join(gt_path,'y'))
    Y_test = [x/255 for x in test_lbl] # normalize between 0 and 1

    pred_test_img = get_image_list(pred_path)
    preds_test = [x/255 for x in pred_test_img] # normalize between 0 and 1

    # Now, we calculate the final test metrics
    iou = []
    for i in range(0, len(preds_test)):
        iou.append( jaccard_index_numpy(Y_test[i], preds_test[i]))
    mean_iou = np.mean(iou)
    print("Test mean IoU:", mean_iou)

    return mean_iou


############################################## ARA

import statistics
import numpy as np
from skimage.measure import label, regionprops_table
import statistics

def morphology_analysis(data, input, delta):
    '''
    The function morphology_analysis returns the mean and median of the area, solidity,
    eccentricity, orientation and number of objects

    Args:
      data: List of labels (binary masks)
      input: List of images associated with the masks
      delta: This value will be added to the factor (initially 1) by which the ratio (ARA) of the image is multiplied.

    Returns:
      The function morphology_analysis returns the mean of the area (and median, only in this case), solidity,
    eccentricity, orientation and number of objects. It returns also the ratio (ARA) value
    '''

    p_area = []
    p_solidity = []
    p_eccentricity = []
    p_orientation = []
    n_objects=[]
    ratio_objects_area=[]
    pixels_to_th= 10
    factor = 1

    label_img = label(data)   # Connected components

    for i in range(label_img.shape[0]):     # for each 2D image

        img = label_img[i]
        area=(label_img.shape[1] * label_img.shape[2])
        if delta != 0.0:
            area = area - np.sum(np.sum((input[i]==0)))

        props = regionprops_table(img, properties=('area', 'solidity', 'eccentricity', 'orientation'))    # Sacar las propiedades

        for i,v in enumerate(props['area']):
            p_area.append(v)
            if v>pixels_to_th:
                ratio = v/area
                ratio = ratio*factor
                ratio_objects_area.append(ratio)

                p_solidity.append(props['solidity'][i])
                p_eccentricity.append(props['eccentricity'][i])
                p_orientation.append(props['orientation'][i])
        n_objects.append(len(np.unique(img))-1)
        factor = factor + delta

    try:
        gt_area_value = statistics.mean(p_area)
        gt_solidity_value = statistics.mean(p_solidity)
        gt_eccentricity_value = statistics.mean(p_eccentricity)
        gt_orientation_value = statistics.mean(p_orientation)
        gt_area_value_median=statistics.median(p_area)
        gt_object_number=statistics.mean(n_objects)
        gt_ratio=statistics.mean(ratio_objects_area)

    except:
        gt_area_value = 0
        gt_solidity_value = 0
        gt_eccentricity_value = 0
        gt_orientation_value = 0
        gt_area_value_median=0
        gt_object_number=0
        gt_ratio=0

    return gt_area_value,gt_solidity_value,gt_eccentricity_value,gt_orientation_value,gt_area_value_median,gt_object_number,gt_ratio

from copy import copy
class CustomSaver():

    def __init__(self, dataset_path, snaps, source, target, path_save='./tmp/'):
        '''
        Args:
          src_data_path: the path to the source dataset
          data_path: the path to the target dataset
          source: the name of source dataset
          target: the name of target dataset.
          analysis_mode: Plots and csv files will be generated and stored with several obtained values
          path_save: the path where the model will be saved, defaults to ARA_tmp_Models (optional). Defaults to
        Models
        '''

        def prepare_data(test_img, test_lbl, use_padding, expand_dims):
            '''
            Given a list of images and labels, it will normalize the images between 0 and 1, and add padding if
            specified

            Args:
              list_img: list of images
              list_lbl: list of labels
              use_padding: If True, add padding to the images (height and width multiple of 256).
              expand_dims: If True, expand the dimensions of the images to add an extra dimension for the number
            of channels.
            '''
            if use_padding:
                test_img = [add_padding(x) for x in test_img]
                test_lbl = [add_padding(x) for x in test_lbl]

            X_test = [x/255 for x in test_img] # normalize between 0 and 1
            Y_test = [x/255 for x in test_lbl] # normalize between 0 and 1

            if expand_dims:
                X_test = np.expand_dims( np.asarray(X_test, dtype=np.float32), axis=-1 ) # add extra dimension
                Y_test = np.expand_dims( np.asarray(Y_test, dtype=np.float32), axis=-1 ) # add extra dimension

            del test_img, test_lbl
            return X_test, Y_test

        self.set = 'test'
        self.domain = target
        self.dataset_path = dataset_path
        test_img, test_lbl = get_xy_image_list(self.dataset_path + self.domain +'/'+ self.set)
        src_test_img, src_test_lbl = get_xy_image_list(self.dataset_path + source +'/'+ 'train_val')

        self.batch_size=1
        self.Xtest, self.Ytest = prepare_data(copy(test_img), test_lbl, use_padding=True, expand_dims=True)
        self.src_Xtest, self.src_Ytest = prepare_data(copy(src_test_img), src_test_lbl, use_padding=False, expand_dims=False)

        self.IoU_test=[]
        self.x=[]
        self.source = source
        self.target = target

        self.area=[]
        self.solidity=[]
        self.eccentricity=[]
        self.orientation=[]
        self.median_area=[]
        self.n_objects=[]
        self.ratio=[]
        self.dif=1000

        self.snaps = snaps
        self.dst_path = path_save
        self.best_model_iou = 0
        self.best_model_epoch = 0

        def get_delta(data, input):
            '''
            Given a set of images, the function returns the ratio of the maximum area of the image to the
            minimum area of the image. Ignoring ceros, almost all padding.

            Args:
              data: List of labels (binary masks)
              input: List of images associated with the masks

            Returns:
              the delta value.
            '''

            area_max = (data[-1].shape[0] * data[-1].shape[1]-np.sum(np.sum((input[-1]==0))))
            area_min = (data[0].shape[0] * data[0].shape[1]-np.sum(np.sum((input[0]==0))))
            delta = (area_max/area_min)/(len(input)-1)
            return delta

        self.src_delta = get_delta(self.src_Ytest, self.src_Xtest) if self.source == 'Kasthuri++' else 0.0

        self.trg_y_noPadding = [remove_padding(self.Ytest[i][:,:,0], x.shape) for i, x in enumerate(test_img)] # remove padding
        self.trg_x_noPadding = [remove_padding(self.Xtest[i][:,:,0], x.shape) for i, x in enumerate(test_img)] # remove padding
        self.trg_y_noPadding = np.asarray(self.trg_y_noPadding)
        self.trg_x_noPadding = np.asarray(self.trg_x_noPadding)
        self.trg_delta = get_delta(self.trg_y_noPadding, self.trg_x_noPadding) if self.target == 'Kasthuri++' else 0.0

        _, self.source_desired_solidity, *_, self.source_desired_ratio = morphology_analysis(np.array(self.src_Ytest), np.array(self.src_Xtest), self.src_delta)

    def get_results(self):
        '''
        The results are stored in a dictionary. The keys are the same as the variable names above

        Returns:
          The dictionary with several features used and obtained during the process.
        '''
        morphology = {}
        morphology['source <{}> delta'.format(self.source)] = self.src_delta
        morphology['target <{}> delta'.format(self.target)] = self.trg_delta
        morphology['src_desired_solidity'] = float(self.source_desired_solidity)
        morphology['best_model'] = str(self.best_model)
        morphology['best_model_iou'] = float(self.best_model_iou)
        morphology['best_model_epoch'] = int(self.best_model_epoch)

        morphology['Epochs'] = np.array(self.x).tolist()
        morphology['IoU'] = np.array(self.IoU_test).tolist()

        morphology['area'] = np.array(self.area).tolist()
        morphology['solidity'] = np.array(self.solidity).tolist()
        morphology['eccentricity'] = np.array(self.eccentricity).tolist()
        morphology['orientation'] = np.array(self.orientation).tolist()
        morphology['Median area'] = np.array(self.median_area).tolist()
        morphology['Object Number'] = np.array(self.n_objects).tolist()
        morphology['Ratio'] = np.array(self.ratio).tolist()

        rm_dir(self.dst_path)
        return morphology

    def make_predictions(self, snap):
        rm_dir(self.dst_path)
        create_dir(self.dst_path)
        set2 = 'train_val' if self.set=='train' else self.set # just for input data dir
        test_command = 'python3 prediction.py ' + \
        '--data-dir-test "' + self.dataset_path + self.domain +'/'+ set2 +'/x" ' + \
        '--data-dir-test-label "' + self.dataset_path + self.domain +'/'+ set2 +'/y" ' + \
        '--data-list-test "' + self.dataset_path + self.domain +'/'+ set2 +'/file_list.txt" ' + \
        '--test-model-path "'+ snap +'" ' + \
        '--num-workers 10 ' + \
        '--gpu 0 ' + \
        '--test_aug 1 ' + \
        '--save-dir "'+ self.dst_path + '" '
        try:
            os.system(test_command)
        except:
            print("++++++++++ Test ERROR ++++++++++")

    def on_epoch_end(self, epoch, snap):
        preds_test = get_image_list(self.dst_path + '/_iter_0') # load
        preds_test = [x/255 for x in preds_test] # normalize between 0 and 1
        preds_test = np.asarray(preds_test) >= 0.5

        iou = []
        for i in range(0, len(preds_test)):
            iou.append( jaccard_index_numpy(self.trg_y_noPadding[i], preds_test[i]) )
        jaccard = float(np.mean(iou))
        print('Jaccard in target: '+ str(jaccard))
        self.IoU_test.append(jaccard)
        self.x.append(int(epoch))

        #target
        gt_area_value,gt_solidity_value,gt_eccentricity_value,gt_orientation_value,gt_area_value_median,gt_object_number,gt_ratio=morphology_analysis(
                preds_test,
                self.trg_x_noPadding,
                self.trg_delta)

        self.area.append(gt_area_value)
        self.solidity.append(gt_solidity_value)
        self.eccentricity.append(gt_eccentricity_value)
        self.orientation.append(gt_orientation_value)
        self.median_area.append(gt_area_value_median)
        self.n_objects.append(gt_object_number)
        self.ratio.append(gt_ratio)
        #Si el ratio es cercano a source_desired_ratio y el número de épocas es superior a N
        act_dif = abs(gt_solidity_value-self.source_desired_solidity)
        if act_dif < self.dif:
            self.dif = act_dif
            self.best_model= snap
            self.best_model_iou = jaccard
            self.best_model_epoch = epoch

    def compute_all(self):
        epoch = 0
        for snap in self.snaps:
            self.make_predictions(snap) # make predictions for test set using current epoch's snap, and store in tmp
            self.on_epoch_end(epoch, snap) # load from tmp and compute morphology and iou
            epoch += 2
        return self.best_model_iou
