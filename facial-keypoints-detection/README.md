# facial keypoints detection

####  Data

<p style='text-align: justify;'> 
Facial keypoints (also called facial landmarks) are the small magenta dots shown on each of the faces in the image above. In each training and test image, there is a single face and 68 keypoints, with coordinates (x, y), for that face. These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc. These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, and so on. Here they are, numbered, and you can see that specific ranges of points match different portions of the face.
</p>

<p align="center">
<img  src="https://github.com/Noob-can-Compile/Facial_Keypoint_Detection/raw/master/images/landmarks_numbered.jpg" alt="keypoints" width="200"/>
</p>

<p style='text-align: justify;'> 
The data is from: https://github.com/rupaai/facial_keypoint_detection
</p>

Here are some sample images in the dataset:

<p align="center">
<img src="images/7.jpg" width=150 height=150><img src="images/8.jpg"  width=150 height=150><img src="images/9.jpg" width=150 height=150><img src="images/10.jpg" width=150 height=150><img src="images/11.jpg" width=150 height=150><img src="images/12.jpg" width=150 height=150>
</p>

#### Model
In this project, EfficientNet-b0 with pretrained weights from ImageNet was used. Model architecture is shown below.

<p align="center">
<img  src="https://1.bp.blogspot.com/-DjZT_TLYZok/XO3BYqpxCJI/AAAAAAAAEKM/BvV53klXaTUuQHCkOXZZGywRMdU9v9T_wCLcBGAs/s640/image2.png" alt="keypoints" width="1200"/>
</p>

<p style='text-align: justify;'>
According google AI news, EfficientNet models achieve both higher accuracy and better efficiency over existing CNNs with less number of parameters. More details seen: https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html
</p>

<p align="center">
<img  src="https://1.bp.blogspot.com/-oNSfIOzO8ko/XO3BtHnUx0I/AAAAAAAAEKk/rJ2tHovGkzsyZnCbwVad-Q3ZBnwQmCFsgCEwYBhgL/s640/image3.png" alt="keypoints" width="450"/>
</p>

####  Model Training:
We used 3462 images for training and 770 images for test and trained the model for 100 epochs with hyperparameters as below:
```
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(params = net.parameters(), lr = 0.001)
scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, verbose=True)
```
<p align="center">
<img  src="loss_curve.png" alt="loss curve" width="450"/>
</p>

And here are some of the keypoints detection results for the test images:
<p align="center">
<img src="outputs/1.png" width=150 height=150><img src="outputs/2.png" width=150 height=150><img src="outputs/3.png" width=150 height=150><img src="outputs/4.png" width=150 height=150><img src="outputs/5.png" width=150 height=150><img src="outputs/6.png" width=150 height=150>
</p>

#### Application
After getting the model, we used the model to predict facial keypoints and add a mask on the face via wrapping.

**1. facial keypoint detection**

Run the face_landmark function in the prediction.ipynb with input image could return this image with keypoints showing.

- imagepath(string): is the path for the input image. Note: the input face should be front face.
- crop(boolean): True is for detect and crop face, False represent using the original image. Use True if the input image is not a head only photo. Default is False

```
face_landmark(imagepath,crop)
```
**2. adding mask to face**

We prepared 7 different types of masks.User can select one of the mask we provided and one image, then we could output the face wearing the mask.

mask library:

<p align="center">
<table><tr><td>Mask 1</td><td>Mask 2</td><td>Mask 3</td><td>Mask 4</td><td>Mask 5</td><td>Mask 6</td><td>Mask 7</td>
 </tr><tr>
    <td><img src="mask/1.jpg" width=100 height=100></td>
    <td><img src="mask/2.jpg"  width=100 height=100></td>
    <td><img src="mask/3.jpg" width=100 height=100></td>
	<td><img src="mask/4.jpg" width=100 height=100></td>
	 <td><img src="mask/5.jpg" width=100 height=100></td>
	 <td><img src="mask/6.jpg" width=100 height=100></td>
	 <td><img src="mask/7.jpg"  width=100 height=100></td>
  </tr></table></p>

Run the face_mask function in the prediction.ipynb with input image and mask id could return this image wearing the mask.

- imagepath(string): is the path for the input image. Note: the input face should be front face.
- mask_id(int): is for choosing a certain mask
- crop(boolean): True is for detect and crop face, False represent using the original image. Use True if the input image is not a head only photo. Default is False

```
face_mask(imagepath,mask_id,crop)
```

**How we did it?**

To map four corners of a mask to one’s nose, lower jaw and two sides of cheek ,we apply homography matrix using SVD. As long as keypoints are well estimated, the wrapping should be beautiful.

1.For normal mask, we match the four point shows in the mask image below with the four point shows in the face below.
<p align="center">
<img src="images/4.png" width=200><img src="images/1.jpg" width=200><img src="images/6.png" width=200></p>

2.We also do this on some cool masks, for example, the one weared by Bane From "Batman:The Dark Knight Rises"

<p align="center">
<img src="images/3.jpg" width=300></p>

For those cool masks without the lowest point, we match the lower jaw with the center bottom point of the masks.

<p align="center">
<img src="images/4.png" width=200><img src="images/2.jpg" width=200><img src="images/5.png" width=200></p>

**Many samples here**
<p align="center">
<img src="outputs/7.png" width=180 height=180><img src="outputs/8.png"  width=180 height=180><img src="outputs/9.png" width=180 height=180></p>
<p align="center">
<img src="outputs/10.png" width=180 height=180><img src="outputs/11.png"  width=180 height=180><img src="outputs/12.png" width=180 height=180></p>

