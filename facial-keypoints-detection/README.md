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

<img src="images/1.jpg" width=100 height=100><img src="images/2.jpg"  width=100 height=100><img src="images/3.jpg" width=100 height=100><img src="images/4.jpg" width=100 height=100><img src="images/5.jpg" width=100 height=100><img src="images/6.jpg" width=100 height=100>

#### Model
In this project, EfficientNet-b0 with pretrained weights from ImageNet was used. Model architecture is shown below.
<p align="center">
<img  src="https://1.bp.blogspot.com/-DjZT_TLYZok/XO3BYqpxCJI/AAAAAAAAEKM/BvV53klXaTUuQHCkOXZZGywRMdU9v9T_wCLcBGAs/s640/image2.png" alt="keypoints" width="1200"/>
</p>
According google AI news, EfficientNet models achieve both higher accuracy and better efficiency over existing CNNs with less number of parameters. More details seen: https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html

<p align="center">
<img  src="https://1.bp.blogspot.com/-oNSfIOzO8ko/XO3BtHnUx0I/AAAAAAAAEKk/rJ2tHovGkzsyZnCbwVad-Q3ZBnwQmCFsgCEwYBhgL/s640/image3.png" alt="keypoints" width="300"/>
</p>

####  Model Training:
We used 3462 images for training and 770 images for test and trained the model for 100 epochs with hyperparameters as below:
```
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(params = net.parameters(), lr = 0.001)
scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, verbose=True)
```
<p align="center">
<img  src="loss_curve.png" alt="loss curve" width="300"/>
</p>

And here are some of the keypoints detection results for the test images:
<img src="result/1.jpg" width=100 height=100><img src="result/2.jpg"  width=100 height=100><img src="result/3.jpg" width=100 height=100><img src="result/4.jpg" width=100 height=100><img src="result/5.jpg" width=100 height=100><img src="result/6.jpg" width=100 height=100>

#### Application
After getting the model, we used the model to predict facial keypoints and add a mask on the face via wrapping.

User can select one of the mask we provided and one image, then we could output the face wearing the mask.

mask library:
<table><tr><td>Mask 1</td><td>Mask 2</td><td>Mask 3</td><td>Mask 4</td><td>Mask 5</td><td>Mask 6</td><td>Mask 7</td>
 </tr><tr>
    <td><img src="mask/1.jpg" width=80 height=80></td>
    <td><img src="mask/2.jpg"  width=80 height=80></td>
    <td><img src="mask/3.jpg" width=80 height=80></td>
	<td><img src="mask/4.jpg" width=80 height=80></td>
	 <td><img src="mask/5.jpg" width=80 height=80></td>
	 <td><img src="mask/6.jpg" width=80 height=80></td>
	 <td><img src="mask/7.jpg"  width=80 height=80></td>
  </tr></table>

- imagepath(string): is the path for the input image. Note: the input face should be front face.
- mask_id(int): is for choosing a certain mask
- crop(boolean): True is for detect and crop face, False represent using the original image. Use True if the input image is not a head only photo. Default is False

```
face_mask(imagepath,mask_id,crop)
```
samples:

<img src="mask/1.jpg" width=200 height=200><img src="mask/2.jpg"  width=200 height=120><img src="mask/3.jpg" width=200 height=200>
<img src="mask/1.jpg" width=200 height=200><img src="mask/2.jpg"  width=200 height=120><img src="mask/3.jpg" width=200 height=200>

