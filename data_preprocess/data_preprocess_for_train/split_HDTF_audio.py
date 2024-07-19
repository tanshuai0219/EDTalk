from pydub import AudioSegment
import os

def split(name,wav_dir,save_dir):
    input_file = os.path.join(wav_dir,name+'.wav')
    output_folder = save_dir


    audio = AudioSegment.from_wav(input_file)


    clip_length = 5 * 1000 
    clips = [audio[i:i+clip_length] for i in range(0, len(audio), clip_length)]


    for i, clip in enumerate(clips):
        os.makedirs(output_folder, exist_ok=True)
        filename = os.path.join(output_folder, f"{name}#{i+1}.wav")

        if len(clip)>3 * 1000:
            clip.export(filename, format="wav")



wav_dir = 'HDTF/crop_video'
save_dir = 'HDTF/split_5s_audio'

wav_list = sorted(os.listdir(wav_dir))
for name in wav_list:
    name = name.split('.')[0]
    print(name)
    split(name,wav_dir,save_dir)