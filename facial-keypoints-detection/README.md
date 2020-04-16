# facial keypoints detection

####  Data

<p style='text-align: justify;'> 
Facial keypoints (also called facial landmarks) are the small magenta dots shown on each of the faces in the image above. In each training and test image, there is a single face and 68 keypoints, with coordinates (x, y), for that face. These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc. These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, and so on. Here they are, numbered, and you can see that specific ranges of points match different portions of the face.
</p>

<p align="center">
<img align="center"  src="https://github.com/Noob-can-Compile/Facial_Keypoint_Detection/raw/master/images/landmarks_numbered.jpg" alt="keypoints" width="200"/>
</p>

<p style='text-align: justify;'> 
The data is from: https://github.com/rupaai/facial_keypoint_detection
</p>

####  Model Training:
loss plot
model result

#### Application
After getting the model, we used the model to predict facial keypoints and add a mask on the face via wrapping.

User can select one of the mask we provided and one image, then we could output the face wearing the mask.

mask library:
<p align="center">
<table><tr><td>Mask 1</td><td>Mask 2</td><td>Mask 3</td><td>Mask 4</td><td>Mask 5</td><td>Mask 6</td><td>Mask 7</td>
 </tr><tr>
    <td><img src="mask/1.jpg" width=80 height=80></td>
    <td><img src="mask/2.jpg"  width=80 height=80></td>
    <td><img src="mask/3.jpg" width=80 height=80></td>
	<td><img src="mask/4.jpg" width=80 height=80></td>
	 <td><img src="mask/5.jpg" width=80 height=80></td>
	 <td><img src="mask/6.jpg" width=80 height=80></td>
	 <td><img src="mask/7.jpg"  width=80 height=80></td>
  </tr></table></center>
  </p>

- imagepath(string): is the path for the input image. Note: the input face should be front face.
- mask_id(int): is for choosing a certain mask
- crop(boolean): True is for detect and crop face, False represent using the original image. Use True if the input image is not a head only photo. Default is False

```
face_mask(imagepath,mask_id,crop)
```
samples:
<p align="center">
<img src="mask/1.jpg" width=200 height=200><img src="mask/2.jpg"  width=200 height=200><img src="mask/3.jpg" width=200 height=200></td>
</p>
<p align="center">
<img src="mask/1.jpg" width=200 height=200><img src="mask/2.jpg"  width=200 height=200><img src="mask/3.jpg" width=200 height=200></td>
</p>
