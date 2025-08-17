import numpy as np
import cv2
import matplotlib.pyplot as plt
from neuroutils.swc.io import load_swc
from neuroutils.marker.io import load_marker
from neuroutils.image.io import load_image
from neuroutils.meta.mapping import extract_neuron_id
from neuroutils.meta.neuron import get_neuron_meta
from neuroutils.image.preprocessor import rescale_image
from neuroutils.config.settings import DEFAULT_RESOLUTION_UNIT
from neuroutils.swc.parser import rescale_swc

# dpi = 300
plt.rcParams['figure.dpi'] = 300

def encoding_projection_direction(projection_direction):
    if projection_direction == 'xy':
        projection_axes = 0
    elif projection_direction == 'xz':
        projection_axes = 1
    elif projection_direction == 'yz':
        projection_axes = 2
    return projection_axes

def get_background_img(img_shape, projection_direction='xy'):
    projection_axes = encoding_projection_direction(projection_direction)
    if projection_axes == 0:
        background_img = np.ones((img_shape[1], img_shape[2])).astype(np.uint8) * 255
    elif projection_axes == 1:
        background_img = np.ones((img_shape[0], img_shape[2])).astype(np.uint8) * 255
    elif projection_axes == 2:
        background_img = np.ones((img_shape[0], img_shape[1])).astype(np.uint8) * 255

    background_img = cv2.cvtColor(background_img, cv2.COLOR_GRAY2RGB)
    return background_img

def plot_img_on_fig(fig, gray_img, projection_direction='xy', alpha=0):
    projection_axes = encoding_projection_direction(projection_direction)
    gray_img = np.max(gray_img, axis=projection_axes)
    # resize
    gray_img = cv2.resize(gray_img, (fig.shape[1], fig.shape[0]))
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    # print(gray_img.shape, fig.shape)

    fig = cv2.addWeighted(fig, alpha, gray_img, 1 - alpha, 0)
    return fig

def plot_swc_on_fig(fig, swc_points, plot_mode="sphere", projection_direction='xy', line_color=(255, 0, 0), line_thickness=1, soma_color=(0, 0, 255), soma_thickness=3):
    projection_axes = encoding_projection_direction(projection_direction)

    for swc_point_id in range(len(swc_points)):
        if(plot_mode == 'sphere'):
            if (projection_axes == 0):
                # cv2.circle(fig, (int(swc_points.iloc[swc_point_id].x), int(swc_points.iloc[swc_point_id].y)), int(swc_points.iloc[swc_point_id].r), line_color, -1)
                cv2.circle(fig, (int(swc_points.iloc[swc_point_id].x), int(swc_points.iloc[swc_point_id].y)), int(swc_points.iloc[swc_point_id].r), line_color, -1)
            elif (projection_axes == 1):
                cv2.circle(fig, (int(swc_points.iloc[swc_point_id].x), int(swc_points.iloc[swc_point_id].z)), int(swc_points.iloc[swc_point_id].r), line_color, -1)
            elif (projection_axes == 2):
                cv2.circle(fig, (int(swc_points.iloc[swc_point_id].y), int(swc_points.iloc[swc_point_id].z)), int(swc_points.iloc[swc_point_id].r), line_color, -1)
        # elif(plot_mode=="line"):
        #     if (projection_axes == 0):
        #         cv2.circle(fig, (int(swc_points.iloc[swc_point_id].x), int(swc_points.iloc[swc_point_id].y)), 1, line_color, -1)
        #     elif (projection_axes == 1):
        #         cv2.circle(fig, (int(swc_points.iloc[swc_point_id].x), int(swc_points.iloc[swc_point_id].z)), 1, line_color, -1)
        #     elif (projection_axes == 2):
        #         cv2.circle(fig, (int(swc_points.iloc[swc_point_id].y), int(swc_points.iloc[swc_point_id].z)), 1, line_color, -1)

        if swc_points.iloc[swc_point_id].parent == -1:
            continue
        swc_point = swc_points.iloc[swc_point_id]
        parent_point = swc_points[swc_points['n'] == swc_point['parent']].iloc[0]
        # print(swc_point, parent_point)

        nx, ny, nz = swc_point.x, swc_point.y, swc_point.z
        px, py, pz = parent_point.x, parent_point.y, parent_point.z

        if (projection_axes == 0):
            cv2.line(fig, (int(nx), int(ny)), (int(px), int(py),), line_color, line_thickness)
        elif (projection_axes == 1):
            cv2.line(fig, (int(nx), int(nz)), (int(px), int(pz),), line_color, line_thickness)
        elif (projection_axes == 2):
            cv2.line(fig, (int(ny), int(nz)), (int(py), int(pz),), line_color, line_thickness)


    if (projection_axes == 0):
        cv2.circle(fig, (int(swc_points.iloc[0].x), int(swc_points.iloc[0].y)), soma_thickness, soma_color, -1)
    elif (projection_axes == 1):
        cv2.circle(fig, (int(swc_points.iloc[0].x), int(swc_points.iloc[0].z)), soma_thickness, soma_color, -1)
    elif (projection_axes == 2):
        cv2.circle(fig, (int(swc_points.iloc[0].y), int(swc_points.iloc[0].z)), soma_thickness, soma_color, -1)

    return fig

def plot_markers_on_fig(fig, markers, projection_direction='xy', markers_color=(0, 255, 0), thickness=3, marker_type='rectangle'):
    projection_axes = encoding_projection_direction(projection_direction)

    for marker_id in range(len(markers)):
        marker = markers.iloc[marker_id]
        if marker_type == 'circle':
            if (projection_axes == 0):
                cv2.circle(fig, (int(marker.x), int(marker.y)), int(marker.radius), (marker.color_b, marker.color_g, marker.color_r), thickness)
            elif (projection_axes == 1):
                cv2.circle(fig, (int(marker.x), int(marker.z)), int(marker.radius), (marker.color_b, marker.color_g, marker.color_r), thickness)
            elif (projection_axes == 2):
                cv2.circle(fig, (int(marker.y), int(marker.z)), int(marker.radius), (marker.color_b, marker.color_g, marker.color_r), thickness)
        elif marker_type == 'rectangle':
            if (projection_axes == 0):
                cv2.rectangle(fig, (int(marker.x - marker.radius), int(marker.y - marker.radius)), (int(marker.x + marker.radius), int(marker.y + marker.radius)), markers_color, thickness)
            elif (projection_axes == 1):
                cv2.rectangle(fig, (int(marker.x - marker.radius), int(marker.z - marker.radius)), (int(marker.x + marker.radius), int(marker.z + marker.radius)), markers_color, thickness)
            elif (projection_axes == 2):
                cv2.rectangle(fig, (int(marker.y - marker.radius), int(marker.z - marker.radius)), (int(marker.y + marker.radius), int(marker.z + marker.radius)), markers_color, thickness)

    return fig

def plot_img_and_swc(img_file, swc_file, save_file,
                     projection_direction='xy',
                     rescale_img_flag=False, rescale_swc_flag=False,
                     flip_y=False, down_sample_flag=False
                     ):
    img = load_image(img_file)
    swc_points = load_swc(swc_file)

    if(rescale_img_flag or rescale_swc_flag):
        neuron_id = extract_neuron_id(swc_file)
        meta = get_neuron_meta(neuron_id)
        xy_resolution = float(meta['xy_resolution'].values[0]) / DEFAULT_RESOLUTION_UNIT
        z_resolution = float(meta['z_resolution'].values[0]) / DEFAULT_RESOLUTION_UNIT
        if rescale_img_flag:
            img = rescale_image(img, xy_resolution=xy_resolution, z_resolution=z_resolution)
        if rescale_swc_flag:
            swc_points = rescale_swc(swc_points, xy_resolution=xy_resolution, z_resolution=z_resolution)
    if(flip_y):
        img = np.flip(img, axis=1)
    if(down_sample_flag):
        img = img[::2, ::2, ::2]

    background = get_background_img(img.shape, projection_direction=projection_direction)

    background = plot_img_on_fig(background, img)
    background = plot_swc_on_fig(background, swc_points, line_color=(255, 0, 0), plot_mode='line')

    plt.imshow(background)
    plt.savefig(save_file)
    plt.close()

def test(img_file, swc_file, marker_file, save_file, projection_direction='xy'):
    img = load_image(img_file)
    swc_points = load_swc(swc_file)
    markers = load_marker(marker_file)

    background = get_background_img(img.shape, projection_direction=projection_direction)

    background = plot_img_on_fig(background, img)
    background = plot_swc_on_fig(background, swc_points, line_color=(255, 0, 0))
    background = plot_markers_on_fig(background, markers)

    plt.imshow(background)
    plt.savefig(save_file)
    plt.close()