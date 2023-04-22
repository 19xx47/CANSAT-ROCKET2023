import streamlit as st
import subprocess

def main():
    st.set_page_config(page_title="Input Video Example")

    st.title("Input Video Example")

    video_file = st.file_uploader("Upload a video", type=["mp4"], key="output_video")

    if video_file is not None:
        st.video(video_file)
        with open("output_video.mp4", "wb") as f:
            f.write(video_file.getbuffer())

        subprocess.run(["python", "imagestovideo.py", "output_video.mp4"])

        st.write("Processing complete.")

        st.video("predictoutput_video.mp4")

if __name__ == "__main__":
    main()
