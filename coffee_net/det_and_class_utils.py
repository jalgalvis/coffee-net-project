#%% import required modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

params = {"ytick.color" : "w",
          "xtick.color" : "w",
          "axes.labelcolor" : "w",
          "axes.edgecolor" : "y",
          "axes.facecolor" : "None",
          "text.color" : 'w'}
plt.rcParams.update(params)
#%%
def display_image_with_boxes(image_to_print, boxes, labels, scores=None, category_index=None, boxes_line=5,
                             max_boxes=None, min_treshhold=0):
    """

    :param image_to_print:image as numpy array
    :param boxes: boxes as numpy array
    :param labels: labels as numpy array
    :param scores: scores as numpy array
    :param category_index: dictionary of labals from classes
    :param boxes_line: thickness of the boxes lines
    :param max_boxes: maximum number of boxes
    :param min_treshhold: treshold of scores to display
    :return:image to display as numpy array
    """
    from object_detection.utils import visualization_utils

    if category_index is None:
        cals = set(list(labels))
        category_index = {i: {'name': i} for i in cals}

    image_to_display = visualization_utils.visualize_boxes_and_labels_on_image_array(
        image=image_to_print,
        boxes=boxes,
        classes=labels,
        scores=scores,
        category_index=category_index,
        use_normalized_coordinates=True,
        line_thickness=boxes_line,
        max_boxes_to_draw=max_boxes,
        min_score_thresh=min_treshhold
    )
    return image_to_display

def adjust_boxes_manual_selection(boxes, MEAN_H, MEAN_W, ROUND_FACTOR_100, MARGIN_H_D,
                                  MARGIN_H_U, MARGIN_W_D, MARGIN_W_U, ROUND_FACTOR_1000):
    # remove outliners (too small and too big boxes)
    boxes_H = boxes[:,2]-boxes[:,0]
    boxes_W = boxes[:,3]-boxes[:,1]
    filtered_ids = ((boxes_H >= MEAN_H * (1-MARGIN_H_D)) & (boxes_H <= MEAN_H * (1+MARGIN_H_U))
                    & (boxes_W >= MEAN_W * (1-MARGIN_W_D)) & (boxes_W <= MEAN_W * (1+MARGIN_W_U)))
    boxes = boxes[filtered_ids]

    # remove boxes in the same place using rounding and filtering unique records
    boxes_anchor_factor = np.round(boxes[:,:2]/ROUND_FACTOR_100, ROUND_FACTOR_1000)
    unique_ids = np.unique(boxes_anchor_factor, axis=0,return_index=True)[1]
    boxes = boxes[unique_ids]

    # checking out average box dimensions
    print(f'initial boxes_H mean = {np.mean(boxes_H)}')
    print(f'initial boxes_H mean = {np.mean(boxes_W)}')
    boxes_H = boxes[:,2]-boxes[:,0]
    boxes_W = boxes[:,3]-boxes[:,1]
    print(f'final boxes_H mean = {np.mean(boxes_H)}')
    print(f'final boxes_H mean = {np.mean(boxes_W)}')
    print(f'boxes = {boxes.shape[0]}')
    return boxes


#%%
def images_from_boxes(original_image, boxes):
    """
    create images from the boxes found in the original image

    :param original_image:original image`
    :param boxes: coordinates to extract images from boxes
    :return: array of boxes
    """

    images = np.empty((0,224,224,3))
    X = original_image.shape[1]
    Y = original_image.shape[0]
    for box in boxes:
        y_min = int(Y * box[0])
        y_max = int(Y * box[2])
        x_min = int(X * box[1])
        x_max = int(X * box[3])
        new_image = tf.image.resize_with_pad(
            image=original_image[y_min : y_max, x_min : x_max],
            target_height=224,
            target_width=224).numpy()
        images = np.vstack((images, new_image[np.newaxis, :]))
    return images

#%%
def classify_coffee_beans(images_boxes):

    """
    function to detect good and bad coffee beans using mobilenet trained on coffee dataset
    :param images_boxes: images to classify
    :return: array with the ids of the classified images
    """

    # loading saved model
    model_mobilenet = tf.keras.models.load_model(filepath='models/mobilenet_for_coffe_class.h5', compile=False)

    #making batch predictions
    predicted_batch = model_mobilenet.predict(images_boxes)
    predicted_batch = tf.squeeze(predicted_batch).numpy()

    #getting ids of the classes
    predicted_ids = np.argmax(predicted_batch, axis=-1)
    predicted_ids = predicted_ids.astype(int)

    return predicted_ids