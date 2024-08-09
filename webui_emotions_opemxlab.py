import gradio as gr
import os


from code_for_webui.download_models_openxlab import download 

# Download models and check for exists
download()

from code_for_webui.demo_EDTalk_A_using_predefined_exp_weights import Demo as Demo_EDTalk_A_using_predefined_exp_weights

from code_for_webui.demo_lip_pose import Demo as Demo_lip_pose

demo_lip_pose = Demo_lip_pose()
demo_EDTalk_A_using_predefined_exp_weights = Demo_EDTalk_A_using_predefined_exp_weights()

def get_example():
    case = [
        [
            'res/results_by_facesr/demo_EDTalk_A.png',
            False,
            "res/results_by_facesr/demo_EDTalk_A.wav",
            "test_data/pose_source1.mp4",
            False,
            "I don't wanna generate emotional expression",
            True,
        ],

        [
            'test_data/identity_source.jpg',
            False,
            "test_data/mouth_source.wav",
            "res/results_by_facesr/demo_EDTalk_A.mp4",
            False,
            "happy",
            True,
        ],
        [
            'test_data/uncrop_face.jpg',
            True,
            "test_data/test/11.wav",
            "test_data/uncrop_Obama.mp4",
            True,
            "surprised",
            True,
        ]
    ]
    return case
    
tips = r"""
### Usage tips of EDTalk
1. If you're not satisfied with the results, check whether the image or the video is cropped.
2. If you're not satisfied with the results, check whether "Use Face Super-Resolution" is selected.
3. If you're not satisfied with the results, choose a pose video with similar head pose to source image.
"""
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
            return gr.Video(value=save_path), gr.Video(value=save_512_path), gr.Markdown(tips)

        else:
            print("else")
            return None, None, gr.Markdown("Video generated failed, please retry it.")
    except:
        return None, None, gr.Markdown("Video generated failed, please retry it.")
    # return f"Output saved to: {save_path}"

def remove_tips():
    return gr.update(visible=False)

def main():



    ### Description
    title = r"""
    <h1 align="center">EDTalk: Efficient Disentanglement for Emotional Talking Head Synthesis</h1>
    """

    description = r"""
    <b>Official 🤗 Gradio demo</b> for <a href='https://github.com/tanshuai0219/EDTalk' target='_blank'><b>EDTalk: Efficient Disentanglement for Emotional Talking Head Synthesis</b></a>.<br>

    How to use:<br>
    1. Upload an image with a face. Ensure the face is not too small and is clearly visible without significant obstructions or blurring.
    2. If the face is not be cropped by crop_image2.py, please click "Crop the Source Image"
    3. Upload a video for head pose source. Ensure the face is not too small and is clearly visible without significant obstructions or blurring.
    4. If the face is not be cropped by crop_video.py, please click "Crop the Pose Video"
    5. Upload an audio
    6. Choose the exp type
    7. (Recommended) click the "Use Face Super-Resolution"
    8. Share the generated videos with your friends and enjoy! 😊
    """

    article = r"""
    ---
    📝 **Citation**
    <br>
    If our work is helpful for your research or applications, please cite us via:
    ```bibtex
    @inproceedings{tan2024edtalk,
    title = {EDTalk: Efficient Disentanglement for Emotional Talking Head Synthesis},
    author = {Tan, Shuai and Ji, Bin and Bi, Mengxiao and Pan, Ye},
    booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
    year = {2024}
    }
    ```
    📧 **Contact**
    <br>
    If you have any questions, please feel free to open an issue or directly reach us out at <b>tanshuai0219@sjtu.edu.cn</b>.
    """

    tips = r"""
    ### Usage tips of EDTalk
    1. If you're not satisfied with the results, check whether the image or the video is cropped.
    2. If you're not satisfied with the results, check whether "Use Face Super-Resolution" is selected.
    3. If you're not satisfied with the results, choose a pose video with similar head pose to source image.
    """

    css = '''
    .gradio-container {width: 85% !important}
    '''

    with gr.Blocks(css=css) as demo:

        # description
        gr.Markdown(title)
        gr.Markdown(description)


        with gr.Row():
            with gr.Column():

                # upload face image
                source_file = gr.Image(type="filepath",label="Select Source Image.")

                crop_image = gr.Checkbox(label="Crop the Source Image")

                driving_audio = gr.Audio(type="filepath", label="Select Audio File")
                pose_video = gr.Video(label="Select Pose Video. If no pose video, generate video will have the default pose")

                crop_video = gr.Checkbox(label="Crop the Pose Video")

                exp_type = gr.Dropdown(
                    choices=["I don't wanna generate emotional expression","angry", "contempt", "disgusted", "fear", "happy", "sad", "surprised"],
                    label="Select Expression Type",
                     value="I don't wanna generate emotional expression"
                    )
                
                face_sr = gr.Checkbox(label="Use Face Super-Resolution")

                submit = gr.Button("Submit", variant="primary")

            with gr.Column():
                output_256 = gr.Video(label="Generated Video (256)")
                output_512 = gr.Video(label="Generated Video (512)")
                output_log = gr.Markdown(label="Usage tips of EDTalk", value=tips ,visible=False)


            submit.click(
                fn=remove_tips,
                outputs=output_log,            
            ).then(
                fn=run_inference,
                inputs=[source_file, crop_image, driving_audio, pose_video, crop_video, exp_type, face_sr],
                outputs=[output_256, output_512, output_log]       
            )

        gr.Examples(
            examples=get_example(),
            inputs=[source_file, crop_image, driving_audio, pose_video, crop_video, exp_type, face_sr],
            run_on_click=True,
            fn=run_inference,
            outputs=[output_256, output_512, output_log],
            cache_examples=True,
        )
        
        gr.Markdown(article)
    demo.launch()

    # # Launch the interface
    # iface.launch()

if __name__ == "__main__":
    main()