# Drowsy-Driver-Detection

# Drowsy Driver Alert

## Environment

## Steps to run
##### 1. Eye Detection
* Run run_extract_eyes.sh<br>
<code>$ sh run_extract_eyes.sh n</code> <br>
&nbsp;&nbsp;&nbsp;&nbsp;where n is the maximum serial number of the video file
* Input: Input/$i/testAlert$i.avi, Input/$i/testDrowsy$i.avi <br>
&nbsp;&nbsp;&nbsp;&nbsp;where $i is a number from 1 to n
* Output: Output/$i/Alert/&ast;.jpg, Output/$i/Drowsy/&ast;.jpg <br>
&nbsp;&nbsp;&nbsp;&nbsp;where $i is a number from 1 to n

* Crop frames (start when successfully detect eyes) for each video (training data for MobileNet)
	* run_extract_eyes.sh line 171: if int(str(frameCounter)) >= 1000:<br>
	&emsp; modify "1000" to change the amount of the cropped image

##### 2. Retrain MobileNet V1 (depth multiplier 1.00)
* Run retrain.sh <br>
<code>$ sh retrain.sh</code>
* Input: retrain/data/train/	(selected image cropped by run_extract_eyes.sh)
* Output: output_graph.pb
* for more imformation, please reference to:
	* [MobileNet V1 (depth multiplier 1.00)](https://www.tensorflow.org/hub/tutorials/image_retraining)
	* comments in retrain_mobile.py

##### 3. Extract Features
* Run extract_features_mobile.py<br>
<code>$ python extract_features_mobile.py</code>

* Input: data/data_file.csv, output_graph.pb
* Output: data/sequences/*.npy

###### def extract_features() (line 66 to line 88)
<ol style="list-style-type:lower-roman;">
<li>line 67 to line 68<ul><li>access data_file.csv</li></ul></li>
<li>line 70 to line 71<ul><li>named the sequence to save</li></ul></li>
<li>line 73 to 75<ul><li>get the training or testing data (ecah includes 26 images) (eg. training/Drowsy/Drowsy_Video1/*.jpg)</li></ul></li>
<li>line 78 to 80<ul><li>read one of the image from the 26 images to get its features</li><li>
features = extractor(image)</li></ul></li>
<li>line 85<ul><li>after extracted features of 26 images, save the sequence></li></ul></li>
<li>line 87<ul><li>append the features to the sequence</li></ul></li>
</ol>

###### def extractor(image_path) (line 42 to 60)
<ol style="list-style-type:lower-roman;">
  <li>line 44 to 47<ul><li>read the pb file and load the retrained model</li></ul></li>
  <li>line 55<ul><li>get the tensor needed to run</li></ul></li>
  <li>line 57<ul><li>get the image data</li><li>
  image_data = read_tensor_from_image_file(image_path)</li></ul></li>
  <li>line 58 to 59<ul><li>obtain the image pooling features through process the image data by the pooling_tensor</li></ul></li>
  <li>line 64<ul><li>return the image pooling features</li></ul></li>
</ol>

###### def read_tensor_from_image_file(file_name,<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;input_height=224, #height of the model requires<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;input_width=224,  #width of the model requires<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;input_mean=0,<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;input_std=255)<br>(line 14 to 40)
* use to convert the data format that is match with Placeholder
		
##### 4. Train and Test
<ol style="list-style-type:lower-alpha;">
  <li>Run data.py and models.py<ul>
      <li><code>$ python data.py<br>$ python models.py</code></li>
      <li>Input: data/sequences/&ast;.npy</li>
      <li>data.py: use to get the data (&ast;.npy)</li>
      <li>models: use to construct the model</li></ul>
  </li>
  <li>Run train.py<ul>
  <li><code>$ python train.py</code></li>
      <li>line 43: X = X.reshape(2, 26, -1)<br>&nbsp;&nbsp;&nbsp;&nbsp;modify "2" according to the amount (Alert+Drowsy) of training data have</li>
      <li>line 45: X_test = X_test.reshape(2, 26, -1)<br>&nbsp;&nbsp;&nbsp;&nbsp;modify "2" according to the amount (Alert+Drowsy) of testing data have</li>
      <li>line 86: nb_epoch = 30<br>&nbsp;&nbsp;&nbsp;&nbsp;modify "30" for how many epoch needed to train</li></ul>
      &nbsp;&nbsp;&nbsp;&nbsp;where the amount of training/testing is according to data_file.csv column[0]
  </li>
</ol>
