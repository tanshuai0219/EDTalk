import gradio as gr
import os


from code_for_webui.download_models_openxlab import download 

# Download models and check for exists
download()

from code_for_webui.demo_EDTalk_A_using_predefined_exp_weights import Demo as Demo_EDTalk_A_using_predefined_exp_weights

from code_for_webui.demo_lip_pose import Demo as Demo_lip_pose

demo_lip_pose = Demo_lip_pose()
demo_EDTalk_A_using_predefined_exp_weights = Demo_EDTalk_A_using_predefined_exp_weights()

def run_inference(source_image, need_crop_source_img, audio_file, pose_video, need_crop_pose_video, exp_type, face_sr):
    # Get file paths
    try:
        source_path = source_image if source_image else ""
        audio_driving_path = audio_file if audio_file else ""
        pose_driving_path = pose_video if pose_video else "test_data/pose_source1.mp4"

        # Construct the command
        if exp_type in ["angry", "contempt", "disgusted", "fear", "happy", "sad", "surprised"]:

            # command = (
            #     f"python demo_EDTalk_A_using_predefined_exp_weights.py --source_path {source_path} "
            #     f"--audio_driving_path {audio_driving_path} --pose_driving_path {pose_driving_path} "
            #     f"--exp_type {exp_type} --save_path {save_path}"
            # )
            demo_EDTalk_A_using_predefined_exp_weights.process_data(source_path, pose_driving_path, audio_driving_path, exp_type, need_crop_source_img, need_crop_pose_video, face_sr)
            save_path = demo_EDTalk_A_using_predefined_exp_weights.run()
        else:
            demo_lip_pose.process_data(source_path, pose_driving_path, audio_driving_path, need_crop_source_img, need_crop_pose_video, face_sr)
            save_path = demo_lip_pose.run()

        save_512_path = save_path.replace('.mp4','_512.mp4')

        # Check if the output video file exists
        if not os.path.exists(save_path):
            return None, gr.Markdown("Error: Video generation failed. Please check your inputs and try again.")
        if face_sr == False:
            print("here face_sr == False:")
            return gr.Video(value=save_path), None, gr.Markdown("Video (256*256 only) generated successfully!")
        
        
        elif os.path.exists(save_512_path):
            print("here os.path.exists(save_512_path):")
            return gr.Video(value=save_path), gr.Video(value=save_512_path), gr.Markdown("Video generated successfully!")

        else:
            print("else")
            return None, None, gr.Markdown("Video generated failed, please retry it.")
    except:
        return None, None, gr.Markdown("Video generated failed, please retry it.")
    # return f"Output saved to: {save_path}"

# Create Gradio interface
iface = gr.Interface(
    fn=run_inference,
    inputs=[
        gr.Image(type="filepath",label="Select Source Image."), # Make sure the image is pre-processed using  crop_image2.py
        gr.Checkbox(label="Crop the Source Image"),
        gr.Audio(type="filepath", label="Select Audio File"),
        gr.Video(label="Select Pose Video."),  #Make sure the video is pre-processed using crop_video.py
        gr.Checkbox(label="Crop the Pose Video"),
        gr.Dropdown(
            choices=["I don't wanna generate emotional expression","angry", "contempt", "disgusted", "fear", "happy", "sad", "surprised"],
            label="Select Expression Type"
        ),
        gr.Checkbox(label="Use Face Super-Resolution")
    ],
    # outputs="text",
    outputs=[gr.Video(label="Generated Video (256)"),
              gr.Video(label="Generated Video (512)"),
              gr.Markdown()],
    title="EDTalk",
    description="Upload the necessary files and parameters to run the inference script."
)

# Launch the interface
iface.launch()
