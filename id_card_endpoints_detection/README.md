# id cards endpoints detection

####  Data

<p style='text-align: justify;'> 
In this project the MIDV-500 dataset was used to train a id-cards detection model. This dataset contains video clips of 50 different identity document types, including 17 types of IDcards, 14 types of passports, 13 types of driving licenses and 6 other identity documents of various countries, and each image was labeled with the four endpoints. We used all the images except passports.
</p>

<p style='text-align: justify;'> 
The data is from: https://arxiv.org/pdf/1807.05786.pdf [MIDV-500: a dataset for identity document analysis and recognition
on mobile devices in video stream]
</p>

Here are some sample images in the dataset:

<p align="center">
<img src="images/1.jpg" width=150 height=280><img src="images/2.jpg"  width=150 height=280><img src="images/3.jpg" width=150 height=280><img src="images/4.jpg" width=150 height=280><img src="images/5.jpg" width=150 height=280><img src="images/6.jpg" width=150 height=280>
</p>

#### Model

In this project, we write a neuron network, model summary are shown below:

<p align="center">
<img src="images/7.png" width=300>
</p>


####  Model Training:

We used 5940 images for training and 660 images for test and trained the model for 100 epochs with hyperparameters as below:
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
After getting the model, we used the model to predict four endpoints of the card and reshape the card to a rectangle via wrapping.

**1. endpoints detection**

Run the prediction function in the prediction.ipynb with input image could return this image with keypoints showing.

- imagepath(string): is the path for the input image. Note: the input face should be front face.

```
prediction(imagepath)
```
sample：
<p align="center">
<img  src="outputs/7.png" alt="output" width="450"/>
</p>

**2. reshape to rectangle**

Run the face_mask function in the prediction.ipynb with input image and mask id could return this image wearing the mask.

- mask_id(int): is for choosing a certain mask

```
face_mask(imagepath,mask_id,crop)
```
sample：
<p align="center">
<img  src="outputs/8.png" alt="output" width="450"/>
</p>
