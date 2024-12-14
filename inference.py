"""Inference on a folder containing an image sequence."""
import sys
sys.path.append('..')

import torch
import argparse
import os
from utils import NUM_CLASSES, CLASS_LIST, CLASS_COLOUR_MAP
from utils import HEATMAP_HOT_0, HEATMAP_HOT_25, HEATMAP_HOT_50, HEATMAP_HOT_75, HEATMAP_HOT_100
from utils import overlay_mask, overlay_output, write_output, load_image, write_image, \
    prepare_deeplabv3_model, prepare_segformer_model, \
    overlay_images, apply_colour_map, create_custom_legend
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import re
import numpy as np

import os
import re
import csv
import numpy as np
from tqdm import tqdm
from utils import CLASS_LIST, CLASS_IDX_MAP
from utils import load_image, write_image, map_integer_encoded_mask

from utilities_segmentation import integer_encoded_mask_to_annotation
from utilities_geometry import get_polygon_centroid, get_polygon_area, polygon_to_region

import torch
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


def get_args_parser():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--arch', default='segformer_b0', type=str,
        choices=['deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large',
                 'segformer_b0', 'segformer_b1', 'segformer_b2', 'segformer_b3', 'segformer_b4', 'segformer_b5'],
        help='Name of architecture to use.')

    # Misc
    parser.add_argument('--input-dir', default='inference_input', type=str, help='Path to images')
    parser.add_argument('--output-dir', default='inference_output', type=str, help='Path to overlaid images')
    parser.add_argument('--models-dir', default='models', type=str, help='Path to models')
    parser.add_argument('--context-window', default=1, type=int, help='x images before and x images after are taken as context window'
                                                                    '0 means training without context window')
    parser.add_argument('--image-size', default=512, type=int, help='Size of images.')
    parser.add_argument('--freeze-backbone', action=argparse.BooleanOptionalAction, default=False,
        help='Whether to load weights from model trained with frozen or finetuned backbone.')
    parser.add_argument('--generate-heatmap', action=argparse.BooleanOptionalAction, default=False,
        help='Whether to generate heatmap.')
    parser.add_argument('--generate-grayscale', action=argparse.BooleanOptionalAction, default=False,
        help='Whether to generate heatmap.')
    parser.add_argument('--generate-overlaid', action=argparse.BooleanOptionalAction, default=True,
        help='Whether to generate heatmap.')
    
    # Postprocessing
    parser.add_argument('--window-size', default=4, type=int, help='Window size when calculating rolling average.')
    parser.add_argument('--relative-threshold', default=0.4, type=float, help='Relative threshold to detect outliers.')

    return parser


def inference(args):
    # ============ inference settings ... ============
    print('============ inference settings ... ============')
    for arg_name in vars(args):
        print(f'{arg_name}: {getattr(args, arg_name)}')   

    # ============ preparing model ... ============
    print('============ preparing model ... ============')
    if 'deeplabv3' in args.arch:
        backbone = args.arch.split('_', maxsplit=1)[1]
        model = prepare_deeplabv3_model(num_classes=NUM_CLASSES,
                                        backbone=backbone,
                                        pretrained=False,
                                        freeze_backbone=False,
                                        context_window=args.context_window)
    elif 'segformer' in args.arch:
        model = prepare_segformer_model(num_classes=NUM_CLASSES,
                                        pretrained_model=args.arch,
                                        pretrained=False,
                                        freeze_encoder=False,
                                        context_window=args.context_window)
    else:
        raise ValueError(f'{args.arch} architecture not implemented')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(os.path.join(args.models_dir, experiment_name, 'model-latest.pth'), map_location=device)
    model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    
    # ============ preparing inputs ... ============
    print('============ preparing inputs ... ============')
    base_input_dir = args.input_dir
    base_output_dir = args.output_dir

    input_dirs = [e.name for e in os.scandir(base_input_dir) if e.is_dir()]
    input_dirs.sort()

    os.makedirs(os.path.join(base_output_dir, experiment_name), exist_ok=True)
    time_of_stoppage_csv_file = open(os.path.join(base_output_dir, experiment_name, 'time_of_stoppage.csv'), 'w', newline='')
    time_of_stoppage_csv_writer = csv.writer(time_of_stoppage_csv_file)
    time_of_stoppage_csv_writer.writerow(['flywell', 'hours', 'days'])

    for input_dir_basename in tqdm(input_dirs):
        input_dir = os.path.join(base_input_dir, input_dir_basename)
        output_dir = os.path.join(base_output_dir, experiment_name, input_dir_basename)

        os.makedirs(output_dir, exist_ok=True)
        if args.generate_heatmap:
            heatmap_dir = os.path.join(output_dir, 'heatmaps')
            os.makedirs(heatmap_dir, exist_ok=True)

        filenames = []
        for filename in os.listdir(input_dir):
            m = re.match(r'^(\d+).*\.jpg$', filename)
            if not m:
                continue
            filenames.append((int(m.groups()[0]), filename))
        filenames.sort()

        # Create a dictionary for faster lookup
        filenames_dict = {order_number: {'image_filename': image_filename,}
                          for (order_number, image_filename) in filenames}

        # Stores for the images and titles to be displayed
        all_images, all_legends, all_titles = [], [], []

        # Information from postprocessing
        csv_path = os.path.join(output_dir, f'{input_dir_basename}.csv')
        csv_file = open(csv_path, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(['image_number', 'filename', 'size', 'centroid_x', 'centroid_y', 'is_fly_detected'])

        for image_number, filename in tqdm(filenames, desc=input_dir_basename):
            # ============ preparing an image sequence ... ===========
            transforms = get_transforms()

            input_tensor = get_image_sequence(input_dir, image_number, filenames_dict, transforms, image_size=args.image_size, context_window=args.context_window)
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

            input_batch = input_batch.to(device)

            # ============ getting predictions ... ============
            with torch.no_grad():
                if 'deeplabv3' in args.arch:
                    output = model.forward(input_batch)['out']
                elif 'segformer' in args.arch:
                    output = model.forward(input_batch)
            output_prediction = output.argmax(1).squeeze()  # output has size of shape '1 c h w', doing argmax at the c (classes) dimension, then squeeze to get 'h w')
            
            # Postprocessing
            raw_mask = output_prediction.cpu().numpy().astype(np.uint8)
            output_prediction, size, centroid = postproccess_mask(raw_mask)

            # ============ writing information to csv file ... ============
            is_fly_detected = 1
            if size == -1:
                is_fly_detected = 0
            writer.writerow([image_number, filename, size, centroid[0], centroid[1], is_fly_detected])
            csv_file.flush()

            # ============ writing output ... ============
            write_output(output_prediction, output_dir, os.path.splitext(filename)[0], scale_info=None, write_grayscale=args.generate_grayscale, write_rgb=False)

            # ============ overlaying output ... ============
            input_image = load_image(os.path.join(input_dir, filename))
            # overlaid_image = overlay_output(input_image, output_predictions, CLASS_LIST, CLASS_COLOUR_MAP)
            overlaid_image = overlay_mask(input_image, output_prediction, CLASS_LIST, CLASS_COLOUR_MAP, alpha=0.9)

            # Write the overlaid image to disk
            if args.generate_overlaid:
                os.makedirs(os.path.join(output_dir, 'overlaid_predictions'), exist_ok=True)
                write_image(os.path.join(output_dir, 'overlaid_predictions', f'overlaid-{os.path.splitext(filename)[0]}.png'), overlaid_image, overwrite=True)

            # ============ generating heatmap ... ============
            if args.generate_heatmap:

                # Take the softmax of the reconstructed prediction (gives softmax heatmap)
                # softmax_heatmap = torch.softmax(
                #     complete_prediction[PRED_SEG_MASK_KEY], dim=0).cpu().numpy()
                softmax_heatmap = torch.softmax(output.squeeze(), dim=0).cpu().numpy()  # add softmax

                # Turn softmax heatmap into images to be displayed per-class in the class list
                for class_name, class_heatmap in zip(CLASS_LIST, softmax_heatmap):
                    if class_name == 'foreground':
                        heatmap = apply_colour_map(class_heatmap, colour_map='hot')

                        # # Add the heatmap to be displayed
                        # all_images.append(heatmap)
                        # all_legends.append(create_custom_legend(
                        #     ['0.00', '0.25', '0.50', '0.75', '1.00'],
                        #     [HEATMAP_HOT_0, HEATMAP_HOT_25, HEATMAP_HOT_50, HEATMAP_HOT_75,
                        #     HEATMAP_HOT_100]))
                        # all_titles.append(f'{class_name} Heatmap')

                        # Add the heatmap overlaid onto the original region to be displayed
                        res = overlay_images(input_image, heatmap, 0.25)
                        all_images.append(res)
                        all_legends.append(None)
                        all_titles.append(f'{class_name} Heatmap OVERLAID')
                
                # Write images to disk
                for data, title in zip(all_images, all_titles):
                    write_image(os.path.join(heatmap_dir, f'{title}-{os.path.splitext(filename)[0]}.png'), data, overwrite=True)
        
        # Postprocess the csv file and create a plot
        plot_sizes_distances(csv_path, input_dir_basename, output_dir, time_of_stoppage_csv_file, time_of_stoppage_csv_writer, window_size=args.window_size, relative_threshold=args.relative_threshold)


def get_image_sequence(images_path, image_number, filenames_dict, transforms, image_size=512, context_window=1):
    order_number = image_number
    order_number_list = [i for i in range(order_number-context_window, order_number+context_window+1)]
    
    train_with_deltas = True

    if not train_with_deltas:
        images = []
        for ord_num in order_number_list:
            if ord_num == order_number:
                image = load_transform_image(images_path, order_number, filenames_dict, transforms, image_size=image_size)
                images.append(image)
            else:
                if ord_num in filenames_dict:
                    image = load_transform_image(images_path, ord_num, filenames_dict, transforms, image_size=image_size)
                else:
                    image = load_transform_image(images_path, order_number, filenames_dict, transforms, image_size=image_size)
                images.append(image)

        image_sequence = torch.cat(images, dim=0)  # shape of ((n x c) x h x w)

        return image_sequence
    else:
        image = load_image(os.path.join(images_path, filenames_dict[order_number]['image_filename']), colour_space='rgb')
        transforms = get_val_transforms()
        data = transforms(image=image)

        # Load images for calculating diffs
        files = []
        for ord_num in order_number_list:
            if ord_num == order_number:
                file = filenames_dict[order_number]['image_filename']
                files.append(os.path.join(images_path, file))
            else:
                if ord_num in filenames_dict:
                    file = filenames_dict[ord_num]['image_filename']
                else:
                    file = filenames_dict[order_number]['image_filename']
                files.append(os.path.join(images_path, file))

        diffs = do_background_subtraction(files)

        # Augment the diffs
        augmented_diffs = [A.ReplayCompose.replay(data['replay'], image=diff, mask=diff)['mask'].permute(2, 0, 1) for diff in diffs]
        images = augmented_diffs.copy()
        # Insert the augmented main image into the middle
        images.insert(args.context_window, data['image'])

        image_sequence = torch.cat(images, dim=0)  # shape of ((n x c) x h x w)
        
        return image_sequence


def load_transform_image(images_path, ord_num, filenames_dict, transforms, image_size):
    record = filenames_dict[ord_num]
    image = load_image(os.path.join(images_path, record['image_filename']), colour_space='rgb')

    # Apply transforms to image
    transformed = transforms(image=image)
    image = transformed['image']

    return image


def get_transforms():
    """
    Transforms to input image.
    """
    transforms = A.Compose([
        A.ToFloat(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transforms


def do_background_subtraction(files):
    diffs = []
    for file_idx in range(len(files)-1):
        im_1 = load_image(files[file_idx])
        im_2 = load_image(files[file_idx + 1])
        diff_im = abs(im_1.astype(np.float32) - im_2.astype(np.float32))

        if diff_im.min() == diff_im.max() == 0:
            diff_im_norm = diff_im.copy()
        else:
            # Normalise between 0 and 1
            diff_im_norm = (diff_im - diff_im.min()) / (diff_im.max() - diff_im.min())

        diffs.append(diff_im_norm)

    return diffs


def polygon_annot_to_data(polygon):
    if len(polygon) < 3:
        return (polygon, (-1, -1), -1, polygon_to_region(polygon))
    return (polygon, get_polygon_centroid(polygon), get_polygon_area(polygon), polygon_to_region(polygon))


def postproccess_mask(mask):
    polygon_annot = integer_encoded_mask_to_annotation(mask=mask, class_idx=1, annotation='polygons', min_sidelength=1)

    polygon_data = list(map(polygon_annot_to_data, polygon_annot))
    if len(polygon_data) == 0:
        return mask, -1, (-1, -1)
    sorted_polygon_data = sorted(polygon_data, key=lambda x: x[2], reverse=True)
    
    largest_polygon = sorted_polygon_data[0]
    size = largest_polygon[2]
    centroid = largest_polygon[1]

    postproccessed_mask = np.zeros_like(mask)
    cv2.fillPoly(postproccessed_mask, [largest_polygon[0]], 1)

    return postproccessed_mask, size, centroid


def longest_not_moving_period(is_moving, n_frames_threshold=5):
    max_length = 0
    current_length = 0
    start_index = -1
    temp_start = -1

    for i, value in enumerate(is_moving):
        if value == 0:  # Object is not moving
            if current_length == 0:  # Start of a new sequence
                temp_start = i
            current_length += 1
        else:  # Object is moving
            if current_length > max_length and current_length >= n_frames_threshold:  # End of a sequence of 0s
                max_length = current_length
                start_index = temp_start
            current_length = 0  # Reset for the next sequence

    # Final check in case the longest sequence is at the end
    if current_length > max_length:
        max_length = current_length
        start_index = temp_start

    return start_index if max_length > 0 else None


def plot_sizes_distances(csv_file_path, flywell_id, output_dir, time_of_stoppage_csv_file, time_of_stoppage_csv_writer, window_size=4, relative_threshold=0.4, distance_threshold=50, n_frames_threshold=5):
    # Load the CSV file
    data = pd.read_csv(csv_file_path)
    data = data[data['size'] != -1].reset_index(drop=True)
    
    # Extract the columns
    image_numbers = data['image_number']
    sizes = data['size']
    
    # Calculate the moving average
    moving_average = sizes.rolling(window=window_size, min_periods=1).mean()
    
    # Calculate the relative change
    relative_change = sizes.pct_change()
    
    # Find indices where the relative change exceeds the threshold
    significant_change_indices = relative_change[abs(relative_change) > relative_threshold].index
    
    # Create a new_sizes series by copying sizes
    new_sizes = sizes.copy()
    
    # Replace the values at significant change indices with the moving average values
    new_sizes[significant_change_indices] = moving_average[significant_change_indices]

    # Add 'is_potential_outlier' and 'adjusted_size' column to the dataframe
    # data['is_potential_outlier'] = 0
    # data.loc[significant_change_indices, 'is_potential_outlier'] = 1

    # data['adjusted_size'] = new_sizes

    # Add distance column
    data['distance'] = np.nan
    data.loc[0, 'distance'] = 0  # Set first distance to 0
    data.loc[1:, 'distance'] = np.sqrt((data['centroid_x'] - data['centroid_x'].shift())**2 + (data['centroid_y'] - data['centroid_y'].shift())**2)

    # Add 'is_moving' column based on threshold
    data['is_moving'] = (data['distance'] > distance_threshold).astype(int)
    
    # Save the updated DataFrame to a new CSV file
    new_csv_file_path = f'{os.path.splitext(csv_file_path)[0]}_postprocessed.csv'
    data.to_csv(new_csv_file_path, index=False, float_format='%.4f')
    
    # Plot the data
    plt.figure(figsize=(12, 8))
    plt.plot(image_numbers, sizes, label='Raw predictions')
    # plt.plot(image_numbers, moving_average, color='red', linestyle='--', label=f'Moving Average (window size: {window_size})')
    # plt.plot(image_numbers, new_sizes, linestyle='-', label='New sizes (adjusted)')

    # Highlight points with significant changes
    # plt.scatter(image_numbers.iloc[significant_change_indices], sizes.iloc[significant_change_indices], color='red', label='Potential outlier')

    # try:
    #     # Define the power function
    #     def exponential_func(x, a, b):
    #         return a * np.exp(b * x)
        
    #     # Fit the power function to the data
    #     params, covariance = curve_fit(exponential_func, image_numbers, sizes)

    #     # Extract parameters
    #     a, b = params

    #     # Plot the fitting curve
    #     plt.plot(image_numbers, exponential_func(image_numbers, a, b), label=f"Fitted Curve: $y={a:.4f}e^{{{b:.4f}x}}$", color='red')
    # except:
    #     pass
    
    # Adding titles and labels
    plt.title(f'{flywell_id}: Size over time')
    plt.xlabel('Frame number')
    plt.ylabel('Size')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    # plt.show()
    
    plt.savefig(os.path.join(output_dir, f'{flywell_id}_postprocessed.png'))
    plt.close()

    # Create a distance plot
    plt.figure(figsize=(12, 8))
    plt.plot(image_numbers, data['distance'], label='Distance')
    time_of_stoppage_index = longest_not_moving_period(data['is_moving'], n_frames_threshold=n_frames_threshold)
    if time_of_stoppage_index:
        x_time_of_stoppage = data.loc[time_of_stoppage_index, 'image_number']
        y_time_of_stoppage = data.loc[time_of_stoppage_index, 'distance']
        plt.scatter(x_time_of_stoppage, y_time_of_stoppage, color='red', label=f'Predicted time of stoppage: {x_time_of_stoppage} hours')
        time_of_stoppage_csv_writer.writerow([flywell_id, x_time_of_stoppage, f'{(x_time_of_stoppage / 24):.4f}'])
    else:
        time_of_stoppage_csv_writer.writerow([flywell_id, None, None])
    time_of_stoppage_csv_file.flush()
    plt.title(f'{flywell_id}: Distance over time')
    plt.xlabel('Frame number')
    plt.ylabel('Distance (pixels)')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, f'{flywell_id}_distance.png'))
    plt.close()


def get_val_transforms():
    """
    Transforms/augmentations for validation images and masks.
    """
    transforms = A.ReplayCompose([
        # ToGrayAndReplicateChannels(),
        A.ToFloat(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transforms


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    # Get the experiment name
    experiment_name = f'{args.arch}_context_window_{args.context_window}_{"frozen" if args.freeze_backbone else "finetuned"}'

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    inference(args)
