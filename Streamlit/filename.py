import streamlit as st
import subprocess
import streamlit as st
import shutil
import os

# CURRENT_THEME = "blue"
# IS_DARK_THEME = True

def main():
    st.image("https://dlscenter.com/wp-content/uploads/2017/06/Barcelona-logo-dream-league-soccer.png", width=100)

    st.set_page_config(page_title="Analyze Land Use From satellite Images")

    st.title("Analyze Land Use From satellite Images")

    video_file = st.file_uploader("Upload a video", type=["mp4"], key="output_video")

    if video_file is not None:
        # Create two columns for displaying the videos side by side
        col1, col2 = st.columns([2, 2])

        # Display the input video in the left column
        with col1:
            video_html = """
                        <video controls width="320" height="240" autoplay="true" muted="true" loop="true">
                        <source 
                        src="./app/static/output_video.mp4" 
                        type="video/mp4" />
                        </video>

                    """
            col1.markdown(video_html, unsafe_allow_html=True)

            with col1:
                video_html = """
                        <video controls width="320" height="240" autoplay="true" muted="true" loop="true">
                        <source 
                        src="./app/static/output_video_that_streamlit_can_play.mp4" 
                        type="video/mp4" />
                        </video>

                    """
                col1.markdown(video_html, unsafe_allow_html=True)
            with col2:
                video_html = """
                        <video controls width="320" height="240" autoplay="true" muted="true" loop="true">
                        <source 
                        src="./app/static/output_video_unsupervies_new.mp4" 
                        type="video/mp4" />
                        </video>

                    """
                col2.markdown(video_html, unsafe_allow_html=True)

        # Display the output video in the right column
        with col2:
            video_html = """
                        <video controls width="320" height="240" autoplay="true" muted="true" loop="true">
                        <source 
                        src="./app/static/output_unsupervisedvideo_chart_new.mp4" 
                        type="video/mp4" />
                        </video>

                    """
            col2.markdown(video_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# col1, col2 = st.columns([2, 2])

# video_html = """
#             <video controls width="320" height="240" autoplay="true" muted="true" loop="true">
#             <source 
#             src="./app/static/output_video.mp4" 
#             type="video/mp4" />
#             </video>

#         """
# col2.markdown(video_html, unsafe_allow_html=True)

# import streamlit as st
# import shutil
# import os
# output_video_unsupervies_new
# st.set_page_config(page_title="Input Video Example")

# st.title("Input Video Example")
# output_video_that_streamlit_can_play
# # Upload input video file
# video_file = st.file_uploader("Upload a video", type=["mp4"], key="output_video")

# # Display the input video file
# if video_file is not None:
#     # Move the uploaded file to ./app/static/
#     file_path = os.path.join(".", "static", "output_video.mp4")
#     with open(file_path, "wb") as f:
#         shutil.copyfileobj(video_file, f)
    
#     # Display the video
#     video_html = f"""
#         <video controls width="320" height="240" autoplay="true" muted="true" loop="true">
#             <source src="./app/static/output_video.mp4" type="video/mp4" />
#         </video>
#     """
#     st.markdown(video_html, unsafe_allow_html=True)
