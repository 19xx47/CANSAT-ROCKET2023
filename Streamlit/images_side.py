import os
import streamlit as st

def main():
    st.title("Analyze Land Use From satellite Images")

    # Get the list of images in the first folder
    folder1 = "/home/worakan/save_images_cansat"
    files1 = os.listdir(folder1)
    files1.sort()

    # Get the list of images in the second folder
    folder2 = "/home/worakan/save_unsupervised"
    files2 = os.listdir(folder2)
    files2.sort()

    folder3 = "/home/worakan/vsave_unsupervised/chart"
    files3 = os.listdir(folder3)
    files3.sort()

    folder4 = "/home/worakan/vsave_unsupervised/output"
    files4 = os.listdir(folder4)
    files4.sort()

    image_files = files1 + files2 + files3 + files4

    # Create a slider to browse through the images
    selected_image = st.sidebar.slider(
        "Select an image",
        min_value=0,
        max_value=len(image_files) - 1,
        step=1
    )
        # Determine which folder the selected image belongs to, and get its path
    if selected_image < len(files1):
        image_path1 = os.path.join(folder1, files1[selected_image])
        image_path2 = os.path.join(folder2, files2[selected_image])
        image_path3 = os.path.join(folder3, files3[selected_image])
        image_path4 = os.path.join(folder4, files4[selected_image])
    else:
        image_path1 = None
        image_path2 = None
        image_path3 = None
        image_path4 = None
    # Display the selected images side-by-side
    col1, col2, col3 ,col4 = st.columns(4)
    if image_path1:
        col1.image(image_path1, use_column_width=True, caption="Folder 1")
    if image_path2:
        col2.image(image_path2, use_column_width=True, caption="Folder 2")
        
    if image_path3:
        col3.image(image_path3, use_column_width=True, caption="Folder 3")
    if image_path4:
        col4.image(image_path4, use_column_width=True, caption="Folder 4")
    # Create two columns for displaying images
    col1, col2, col3 ,col4 = st.columns(4)
# Initialize selected_image1 and selected_image2 to display the first images in each folder by default
    selected_image1 = 0
    selected_image2 = 0
    selected_image3 = 0
    selected_image4 = 0

    # Display an image from the first folder in the left column
    if files1:
        selected_image1 = st.sidebar.selectbox("Select an image from folder 1", ["None"] + files1)
        if selected_image1 != "None":
            image_path1 = os.path.join(folder1, selected_image1)
            col1.image(image_path1, use_column_width=True, caption="Folder 1")

    # Display an image from the second folder in the right column
    if files2:
        selected_image2 = st.sidebar.selectbox("Select an image from folder 2", ["None"] + files2)
        if selected_image2 != "None":
            image_path2 = os.path.join(folder2, selected_image2)
            col2.image(image_path2, use_column_width=True, caption="Folder 2")

    if files3:
        selected_image3 = st.sidebar.selectbox("Select an image from folder 3", ["None"] + files3)
        if selected_image3 != "None":
            image_path3 = os.path.join(folder3, selected_image3)
            col3.image(image_path3, use_column_width=True, caption="Folder 3")
    
    if files4:
        selected_image4 = st.sidebar.selectbox("Select an image from folder 4", ["None"] + files4)
        if selected_image4 != "None":
            image_path4 = os.path.join(folder4, selected_image4)
            col4.image(image_path4, use_column_width=True, caption="Folder 4")
    # # Display an image from the first folder in the left column
    # if files1:
    #     selected_image1 = st.sidebar.selectbox("Select an image from folder 1", files1)
    #     image_path1 = os.path.join(folder1, selected_image1)
    #     col1.image(image_path1, use_column_width=True, caption="Folder 1")

    # # Display an image from the second folder in the right column
    # if files2:
    #     selected_image2 = st.sidebar.selectbox("Select an image from folder 2", files2)
    #     image_path2 = os.path.join(folder2, selected_image2)
    #     col2.image(image_path2, use_column_width=True, caption="Folder 2")

if __name__ == "__main__":
    main()
