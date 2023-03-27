
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from io import BytesIO
import streamlit_toggle as tog

def reshape_img(img, size):
    '''
    Reshapes an PIL.Image to a given size in pixels
    while keeping the aspect ratio intact (center 
    crops the image if necessary)
    '''
    aspect_img = np.shape(img)[0]/np.shape(img)[1]
    aspect_size = size[0]/size[1]
    w, h = img.size
    if aspect_size > aspect_img:
        img = img.crop((w/2-h/aspect_size/2, 0, w/2+h/aspect_size/2, h))
    elif aspect_size < aspect_img:
        img = img.crop((0, h/2-w*aspect_size/2, w, h/2+w*aspect_size/2))
    img = img.resize(np.flip(size))
    return np.array(img)

def create_image_merge(images, pattern, output_dims = (1000,1000), out_path = None, threshold = 100):
    
    # reshape pattern to output dimensions
    pattern = reshape_img(pattern, output_dims)

    # convert pattern to 2 dimensional grey scale image
    if len(np.shape(pattern))!=2:
        grey_scale_pattern = np.mean(pattern, axis=2)
    else:
        grey_scale_pattern = pattern

    # reshape the input images to output dimensions
    img_1 =  reshape_img(ImageOps.exif_transpose(images[0]), output_dims)
    img_2 = reshape_img(ImageOps.exif_transpose(images[1]), output_dims)

    # binarize the pattern
    layer_1 = grey_scale_pattern > threshold
    layer_2 = grey_scale_pattern <= threshold 

    # create an image filter from the pattern
    l1 = np.repeat(layer_1.reshape(np.shape(pattern)[0], np.shape(pattern)[1],1),3,axis=2)
    l2 = np.repeat(layer_2.reshape(np.shape(pattern)[0], np.shape(pattern)[1],1),3,axis=2)

    # create the new image
    new_image = img_1*l2 + img_2*l1

    return new_image
   
if __name__ == '__main__':

    st.title("Pattern-based Image Merging")

    # uploading the necessary images
    pattern_file_buffer = st.file_uploader('Upload a black and white pattern as png, jpeg, webp here.') # Extend to multi colors (with input on how many? in that case kmeans)
    img_file_buffer = st.file_uploader('Upload two images that you want to merge. By default, the first will replace the white and the second the black areas in your pattern image. This behavior can be reversed with the switch toggle that will appear below on the right after upload.', accept_multiple_files=True) # what if only one is given? Or too many?

    # merging the images
    if img_file_buffer and pattern_file_buffer:
        image_from_file = [Image.open(i).convert('RGB') for i in img_file_buffer]
        pattern = Image.open(pattern_file_buffer).convert('RGB')
        width = st.slider('Width of resulting image in pixel:', min_value=50, max_value=2000, value=1000, step=10)
        height = st.slider('Height of resulting image in pixel:', min_value=50, max_value=2000, value=1000, step=10)
        inverted = tog.st_toggle_switch(label="Change order of images", 
                    key="invert", 
                    default_value=False, 
                    label_after = False, 
                    inactive_color = '#D3D3D3', 
                    active_color="#11567f", 
                    track_color="#29B5E8"
                    )
        if inverted:
            image_from_file.reverse()
        if st.button("merge images", type="primary"):      
            new_image = create_image_merge(image_from_file, pattern, output_dims=(width, height))
            plt.imshow(new_image)
            plt.axis('off')
            fig = plt.gcf()
            st.pyplot(fig)

            buf = BytesIO()
            img = Image.fromarray(new_image)
            img.save(buf, format="JPEG")
            byte_im = buf.getvalue()
            btn = st.download_button(
                label="Download Image",
                data=byte_im,
                file_name="merged.jpeg",
                mime="image/jpeg",
                )

