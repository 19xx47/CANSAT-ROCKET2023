import streamlit as st
import subprocess

def main():
    st.set_page_config(page_title="Input Video Example")

    st.title("Input Video Example")

    video_file = st.file_uploader("Upload a video", type=["mp4"], key="output_video")

    if video_file is not None:
        # Create two columns for displaying the videos side by side
        col1, col2, col3, col4 = st.columns(4)

        # Display the input video in the left column
        with col1:
            st.video(video_file)
            with open("output_video.mp4", "wb") as f:
                f.write(video_file.getbuffer())

        # Convert the input video to output video and display it in the right column
        with col2:
            # subprocess.run(["python", "imagestovideo.py", "output_video.mp4"])

            st.write("Processing complete.")

            video_file = open('output_video_that_streamlit_can_play.mp4', 'rb') #enter the filename with filepath

            # video_bytes = video_file.read() #reading the file

            st.video(video_file) #displaying the video

            # st.video("output_video_that_streamlit_can_play.mp4")
        with col3:
                # subprocess.run(["python", "./STEGO/src/unsuoervised.py", "output_video.mp4"])
                st.write("2nd Processing  complete.")
                video_file = open('output_video_unsupervies_new.mp4', 'rb') 
                st.video(video_file) 

        with col4:
                st.write("3rd Processing  complete.")
                video_file = open('output_unsupervisedvideo_chart_new.mp4', 'rb') 
                st.video(video_file)
                
if __name__ == "__main__":
    main()

    col1, col2, col3, col4 = st.columns(4)
