import sys
import lmdb
from PIL import Image
from io import BytesIO

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

from os import listdir, path

if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
	raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
							before running this script!')

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob

import face_detection
hdtf_save_dir = 'HDTF/bbox'
mead_save_dir = 'MEAD_front/bbox'
parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=2, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)

args = parser.parse_args()

fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
									device='cuda:{}'.format(id)) for id in range(args.ngpu)]


env = lmdb.open(
	'EDTalk_lmdb',
	max_readers=32,
	readonly=True,
	lock=False,
	readahead=False,
	meminit=False,
)

def process_video_file(vfile, args, gpu_id):
	splits = vfile.split('#')
	if len(splits)==2:
		save_path = os.path.join(hdtf_save_dir, vfile+'.npy')

	else:
		save_path = os.path.join(mead_save_dir, vfile+'.npy')

	print(vfile)

	if os.path.exists(save_path):
		return 	
	frames = []


	with env.begin(write=False) as txn:
		key = format_for_lmdb(vfile, 'length')
		length = int(txn.get(key).decode('utf-8'))
		for j in range(length):
			key = format_for_lmdb(vfile, j) # M027#neutral#014-0000153
			img_bytes = txn.get(key)
			img = Image.open(BytesIO(img_bytes))

			img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
			frames.append(img)
	


	batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]

	i = -1

	bbox = []

	for fb in batches:
		preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))

		for j, f in enumerate(preds):
			i += 1
			if f is None:
				continue

			x1, y1, x2, y2 = f
			bbox.append(np.array(f))
			# cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), fb[j][y1:y2, x1:x2])
	bbox = np.array(bbox)
	np.save(save_path, bbox)



def mp_handler(job):
	vfile, args, gpu_id = job
	# try:
	process_video_file(vfile, args, gpu_id)
	# except KeyboardInterrupt:
	# 	exit(0)
	# except:
	# 	traceback.print_exc()
import json
def main(args):
	print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))

	# filelist = glob(path.join(args.data_root, '*/*.mp4'))
	filelist = []
	with open('data_preprocess/lists/train.json',"r") as f:
		filelist += json.load(f)
	with open('data_preprocess/lists/test.json',"r") as f:
		filelist += json.load(f)

	filelist = sorted(filelist)
	jobs = [(vfile, args, i%args.ngpu) for i, vfile in enumerate(filelist)]
	p = ThreadPoolExecutor(args.ngpu)
	futures = [p.submit(mp_handler, j) for j in jobs]
	_ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]


def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')
if __name__ == '__main__':
	main(args)



	if not env:
		raise IOError('Cannot open lmdb dataset', 'EDTalk_lmdb')



