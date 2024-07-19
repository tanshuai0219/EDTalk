import os
from moviepy.editor import VideoFileClip

def split_video(input_file, duration=5, name=None):

    video = VideoFileClip(input_file)
    total_duration = video.duration

    start_time = 0
    end_time = duration
    count = 1

    while start_time < total_duration:
        if end_time > total_duration:
            end_time = total_duration
        nnn = name+'#'+str(count)
        output_file = os.path.join('HDTF/split_5s_video', nnn+'.mp4')
        sub_video = video.subclip(start_time, end_time)
        sub_video.write_videofile(output_file)

        start_time += duration
        end_time += duration
        count += 1

    video.close()


import os
import json, glob


root_dir = 'HDTF/video'
videos = glob.glob1(root_dir,'*.mp4')

for v in videos:
    video_path = os.path.join(root_dir, v)
    split_video(video_path, 5, v.split('.')[0])