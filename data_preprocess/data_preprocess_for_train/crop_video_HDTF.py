import face_alignment
import skimage.io
import numpy
from argparse import ArgumentParser
from skimage import img_as_ubyte
from skimage.transform import resize
from tqdm import tqdm
import os, glob
import imageio
import numpy as np
import warnings, json
warnings.filterwarnings("ignore")
import oss2
import pdb

import multiprocessing
from functools import partial

HDTF_dir = 'HDTF/original_videos'
save_HDTF_dir = 'HDTF/video'


all_data = sorted(os.listdir(HDTF_dir))
print(len(all_data))


def extract_bbox(frame, fa):
    if max(frame.shape[0], frame.shape[1]) > 640:
        scale_factor =  max(frame.shape[0], frame.shape[1]) / 640.0
        frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
        frame = img_as_ubyte(frame)
    else:
        scale_factor = 1
    frame = frame[..., :3]
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    if len(bboxes) == 0:
        return []
    return np.array(bboxes)[:, :-1] * scale_factor



def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def join(tube_bbox, bbox):
    xA = min(tube_bbox[0], bbox[0])
    yA = min(tube_bbox[1], bbox[1])
    xB = max(tube_bbox[2], bbox[2])
    yB = max(tube_bbox[3], bbox[3])
    return (xA, yA, xB, yB)


def compute_bbox(start, end, fps, tube_bbox, frame_shape, inp,outp, image_shape, increase_area=0.1):
    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top

    #Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
    h, w = bot - top, right - left

    start = start / fps
    end = end / fps
    time = end - start

    scale = f'{image_shape[0]}:{image_shape[1]}'

    return f'ffmpeg -i {inp} -ss {start} -t {time} -filter:v "crop={w}:{h}:{left}:{top}, scale={scale}" {outp} -y'


def compute_bbox_trajectories(trajectories, fps, frame_shape, args):
    commands = []
    for i, (bbox, tube_bbox, start, end) in enumerate(trajectories):
        if (end - start) > args.min_frames:
            command = compute_bbox(start, end, fps, tube_bbox, frame_shape, inp=args.inp,outp = args.outp, image_shape=args.image_shape, increase_area=args.increase)
            commands.append(command)
    return commands

device = 'cuda'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
def process_video(args):

    video = imageio.get_reader(args.inp)

    trajectories = []
    previous_frame = None
    fps = video.get_meta_data()['fps']
    commands = []
    try:
        for i, frame in tqdm(enumerate(video)):
            frame_shape = frame.shape # (1080, 1920, 3)
            bboxes =  extract_bbox(frame, fa)
            ## For each trajectory check the criterion
            not_valid_trajectories = []
            valid_trajectories = []

            for trajectory in trajectories:
                tube_bbox = trajectory[0]
                intersection = 0
                for bbox in bboxes:
                    intersection = max(intersection, bb_intersection_over_union(tube_bbox, bbox))
                if intersection > args.iou_with_initial:
                    valid_trajectories.append(trajectory)
                else:
                    not_valid_trajectories.append(trajectory)

            commands += compute_bbox_trajectories(not_valid_trajectories, fps, frame_shape, args)
            trajectories = valid_trajectories

            ## Assign bbox to trajectories, create new trajectories
            for bbox in bboxes:
                intersection = 0
                current_trajectory = None
                for trajectory in trajectories:
                    tube_bbox = trajectory[0]
                    current_intersection = bb_intersection_over_union(tube_bbox, bbox)
                    if intersection < current_intersection and current_intersection > args.iou_with_initial:
                        intersection = bb_intersection_over_union(tube_bbox, bbox)
                        current_trajectory = trajectory

                ## Create new trajectory
                if current_trajectory is None:
                    trajectories.append([bbox, bbox, i, i])
                else:
                    current_trajectory[3] = i
                    current_trajectory[1] = join(current_trajectory[1], bbox)


    except IndexError as e:
        raise (e)

    commands += compute_bbox_trajectories(trajectories, fps, frame_shape, args)
    return commands

def oss_upload(bucket, oss_pth, local_pth, try_times=3):
    is_success = False
    for i in range(try_times):
        try:
            bucket.put_object_from_file(oss_pth, local_pth)
            is_success = True
            break
        except:
            continue
    return is_success

def oss_download(bucket, oss_pth, local_pth, try_times=3):
    is_success = False
    for i in range(try_times):
        try:
            bucket.get_object_to_file(oss_pth, local_pth)
            is_success = True
            break
        except:
            continue
    return is_success



def process_single_video(name, args):
    prefix = name.split('.')[0]
    source_path = os.path.join(HDTF_dir, name)
    target_path = os.path.join(save_HDTF_dir, prefix + '.mp4')
    if os.path.exists(target_path):
        return None

    args.inp = source_path
    args.outp = target_path

    try:
        commands = process_video(args)

        for command in commands:
            print({prefix:command})
            os.system(command)

    except:
        return None

import glob
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape")
    parser.add_argument("--increase", default=0.1, type=float, help='Increase bbox by this amount')
    parser.add_argument("--iou_with_initial", type=float, default=0.25, help="The minimal allowed iou with inital bbox")
    parser.add_argument("--inp", default=None, help='Input image or video')
    parser.add_argument("--outp", default=None, help='Input image or video')
    parser.add_argument("--min_frames", type=int, default=0,  help='Minimum number of frames')
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")


    args = parser.parse_args()
    print(args.cpu)
    cmds = []

    num_processes = 5

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=num_processes) as pool:
        # Use partial to pass the args parameter to the function
        func = partial(process_single_video, args=args)
        # Map the function over the list of video names
        results = pool.map(func, all_data)

