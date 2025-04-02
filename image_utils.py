import io
from PIL import Image
import PIL
import cv2
import numpy as np
import base64
import urllib
from typing import Union
import torch
# from ml.humanAI.app.upscale import UpscaleImage
from PIL import Image, ImageOps


# def read_image(image: str|bytes, isCV2: bool = False, scale="BGR"):  
def read_image(image: Union[str,bytes], isCV2: bool = False, scale:str="BGR"):  


    if isCV2:
        if image.startswith("http"):  # Read image from URL
            # Download the image using urllib
            with urllib.request.urlopen(image) as response:
                img_array = np.array(bytearray(response.read()), dtype=np.uint8)

            # Decode the image using OpenCV
            image = cv2.imdecode(img_array, -1)  # Consider -1 flag for transparency

            # req = urllib.urlopen(image)
            # arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            # image = cv2.imdecode(arr, -1) # 'Load it as it is'

        elif len(image) > 455: #isinstance(image, bytes):  # Read image from Byte string  (E.g. b'asfasd')
            # Remove potential header (everything before the first ',')
            header_end_index = image.find(",")
            if header_end_index != -1:
                image = image[header_end_index + 1:]  # Slice to remove header
            # Convert the image string to a byte string
            # byte_str = bytes(image, 'utf-8')
            byte_str = image.encode()

            # Decode the byte string from base64
            decoded_str = base64.b64decode(byte_str)

            # Convert the decoded byte string to a numpy array
            np_arr = np.frombuffer(decoded_str, np.uint8)

            # Decode the numpy array to an OpenCV image
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        else:       # Read image from file path
            image = cv2.imread(image)
        
        if scale == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    else:
    
        if image.startswith("http"):  # Read image from URL
            
            with urllib.request.urlopen(image) as response:
                image_file = response.read()            
            image_file = io.BytesIO(image_file)
            image = Image.open(image_file)

        elif len(image)>455 : #isinstance(image, bytes):  # Read image from Byte string
            # image = Image.open(io.BytesIO(image))
            # Convert the image string to a byte string
            byte_str = bytes(image, 'utf-8')

            # Decode the image string
            img_bytes = base64.b64decode(byte_str)

            # Create a BytesIO object
            img_buf = io.BytesIO(img_bytes)

            # Open the image as a PIL Image object
            image = Image.open(img_buf)
        else:
            image = Image.open(image)   # Read image from file path

        if scale == "RGB":
            image = image.convert('RGB')
   
    return image

def detect_and_crop(image, candidate, apparel_type='top'):
    """
    Detects keypoints in an image and crops it based on the provided candidate and apparel type.
    Refer to this image for the keypoint indices: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
    or this image: https://school.codelink.ai/blog/data-science/algorithm-kalidokit/img/6.webp
    Parameters:
    - image: The input image to be cropped.
    - candidate: List containing the candidate keypoints.
    - apparel_type: A string specifying the type of apparel ('top' or 'bottom'). Default is 'top'.

    Returns:
    - cropped_image: The cropped image based on the detected keypoints and apparel type.
    """
    # Assuming the keypoint is at index 0

    # If the apparel type is 'top', crop the image from the top of the keypoint
    if apparel_type == 'top':
        if len(candidate[14]) == 4:
            x, y, c, n = candidate[14]  # Extract the coordinates of the keypoint
        else:
            x, y = candidate[14]  # Extract the coordinates of the keypoint
        
        crop = [int(y), int(image.shape[1])]  # Calculate the crop dimensions
        # th, tw = [int(round(a / 2)) for a in crop]  # Calculate the half width and half height
        # cropped_image = image[0:crop[0], 0:crop[1]]  # Crop the image
        # cropped_image = image[0:int(y*image.shape[0]),:]
        cropped_image = image
    else:
        return image
    
    return cropped_image


def image_to_b64s(image, isBW = False)-> str:
   
    if isinstance(image, cv2.UMat)  or isinstance(image, np.ndarray) : # or isinstance(image, np.array)
        if not isBW:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Encode the image as JPG format
        _, buffer = cv2.imencode('.jpg', image)

        # Convert the binary data to base64 encoding
        jpg_as_text = base64.b64encode(buffer)

        return jpg_as_text.decode()
    elif isinstance(image,PIL.Image.Image) or isinstance(image, PIL.WebPImagePlugin.WebPImageFile):
        image.convert('RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        jpg_as_text = base64.b64encode(buffered.getvalue())

        return jpg_as_text.decode()

from torchvision import transforms
def remove_bg(image):
    # pip install kornia timm
    # image is pil

    # Load BiRefNet with weights
    from transformers import AutoModelForImageSegmentation
    birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
    torch.set_float32_matmul_precision(['high', 'highest'][0])
    birefnet.to('cuda')
    birefnet.eval()
    transform_image = transforms.Compose([
        transforms.Resize(image.size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_image = transform_image(image).unsqueeze(0).to('cuda')

    # Remove background
    with torch.no_grad():
        output = birefnet(input_image)[-1].sigmoid().cpu()
    # image = output['out'].squeeze().permute(1, 2, 0).numpy() * 255
    
    pred = output[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)

    return image

# ui = UpscaleImage(scale=4)

def resize_if_necessary(image):
    max_width = 764
    max_height = 1024

    # image = Image.open(image_path)
    
    width, height = image.size

    # Check if resizing is necessary
    if width > max_width or height > max_height:
        # Calculate the new size preserving the aspect ratio
        aspect_ratio = width / height
        
        if width > max_width:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_width = width
            new_height = height
        
        if new_height > max_height:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
        
        resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
        return resized_image
    
    # Return the original image if no resizing is necessary
    return image

def improve_resolution(image, return_pil=False, rem_image=None,size=(768, 1024)):
    # image is a cv2 instance
    
    if isinstance(image, PIL.Image.Image):            
        # if rem_bg:
        #     # br = BackgroundRemover()
        #     # image = cv2.cvtColor(br(image,isCV2=True), cv2.COLOR_BGR2RGB)
        #     image = remove_bg(image)
        # convert pil to cv2
        image = np.array(image)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if rem_image is None:
        rem_image = image
    # remove_bg(image)

    original = image.copy()
    W, H = rem_image.shape[1], rem_image.shape[0]

    gray = cv2.cvtColor(rem_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Obtain bounding rectangle and extract ROI
    x,y,w,h = cv2.boundingRect(array=thresh)
    
    # cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)    
    ROI = original[max(0,y-10):min(y+h+10,H), max(0,x-50):min(x+w+50, W)]
    # ROI = original[max(0,y-10):min(y+h+10,H), 0:W]

    # Add alpha channel    
    b,g,r = cv2.split(ROI)
    alpha = np.ones(b.shape, dtype=b.dtype) * 50
    # ROI = cv2.merge([b,g,r,alpha])
    ROI = cv2.merge([r,g,b,alpha])
    ROI = cv2.cvtColor(ROI, cv2.COLOR_RGBA2BGR)
    # Improve resolution of ROI
    ROI_pil = Image.fromarray(ROI) #.convert('RGB')
    max_width, max_height = size
    width, height = ROI_pil.size

    # if width < max_width and height < max_height:
    #     ROI_upscaled = ui(ROI_pil)
    # else:
    ROI_upscaled = ROI_pil

    ROI_upscaled.thumbnail(size=size, resample=Image.Resampling.LANCZOS)  # HAMMING

    ROI_upscaled = add_padding(ROI_upscaled=ROI_upscaled)  

    if  return_pil:
        return ROI_upscaled

    # Convert PIL to cv2
    open_cv_image = np.array(ROI_upscaled)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    return open_cv_image

def add_padding(ROI_upscaled, size=(768, 1024)):
    desired_width, desired_height = size

    width, height = ROI_upscaled.size

    # Extract the background color (assuming it is the color of the top-left pixel)
    background_color = ROI_upscaled.getpixel((0, 0))

    # Calculate padding
    if width < desired_width:
        padding_left = (desired_width - width) // 2
        padding_right = desired_width - width - padding_left
    else:
        padding_left = 0
        padding_right = 0

    if height < desired_height:
        padding_top = (desired_height - height) // 2
        padding_bottom = desired_height - height - padding_top
    else:
        padding_top = 0
        padding_bottom = 0

    # Add padding
    padding = (padding_left, padding_top, padding_right, padding_bottom)
    padded_image = ImageOps.expand(ROI_upscaled, border=padding, fill=background_color)

    return padded_image

def update_resolution(image, target_resolution=(1600,900)): # PIL Image
    
    image.thumbnail(size=target_resolution, resample=Image.Resampling.LANCZOS)  # HAMMING - Resize image to closest dimension
    image_size = image.size
    print(image_size)
    width = image_size[0]
    height = image_size[1]

    if(target_resolution[0] != height or target_resolution[1] != width):
        
        background = Image.new('RGBA', target_resolution, (207,203,199,255))
        offset = (int(round(((target_resolution[1]  - width) / 2), 0)), int(round(((target_resolution[0]  - height) / 2),0)))
        background.paste(image, offset)
        print(background.size)
        print("Image has been resized !")
        return background
        
    else:
        print("Image is already in proportion, it has not been resized !")
        return image