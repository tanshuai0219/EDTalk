import gradio as gr
import os

def run_inference(source_image, need_crop_source_img, audio_file, pose_video, need_crop_pose_video, exp_type, save_path, face_sr):
    # Get file paths
    source_path = source_image if source_image else ""
    audio_driving_path = audio_file if audio_file else ""
    pose_driving_path = pose_video if pose_video else ""

    # Construct the command
    command = (
        f"python demo_EDTalk_A_using_predefined_exp_weights.py --source_path {source_path} "
        f"--audio_driving_path {audio_driving_path} --pose_driving_path {pose_driving_path} "
        f"--exp_type {exp_type} --save_path {save_path}"
    )

    # crop_source_img if checked

    if need_crop_source_img:
        command += " --need_crop_source_img"

    if need_crop_pose_video:
        command += " --need_crop_pose_video"


    # Add the face_sr flag if checked
    if face_sr:
        command += " --face_sr"
    
    print(command)  # For debugging, you can remove this line later
    # Run the command
    os.system(command)
    return f"Output saved to: {save_path}"

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
            choices=["angry", "contempt", "disgusted", "fear", "happy", "sad", "surprised"],
            label="Select Expression Type"
        ),
        gr.Textbox(label="Enter Output Path (e.g. c:\edtalk\output\\video.mp4)"),
        gr.Checkbox(label="Use Face Super-Resolution")
    ],
    outputs="text",
    title="EDTalk (emotions video)",
    description="Upload the necessary files and parameters to run the inference script."
)

# Launch the interface
iface.launch()
