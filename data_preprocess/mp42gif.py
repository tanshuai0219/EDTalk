from moviepy.editor import VideoFileClip

def mp4_to_gif(mp4_path, gif_path):
    # Load the video file
    clip = VideoFileClip(mp4_path)

    # Write the GIF file
    clip.write_gif(gif_path)

    # Close the video file
    clip.close()

# Example usage
mp4_path = "/data/ts/code/EDTalk/res/results_by_facesr/demo_lip_pose5_512.mp4"
gif_path = "/data/ts/code/EDTalk/res/results_by_facesr/demo_lip_pose5_512.gif"
mp4_to_gif(mp4_path, gif_path)
