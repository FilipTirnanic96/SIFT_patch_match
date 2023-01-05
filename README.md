# Patch match algorithm
## Problem statement
<p align="justify">

Our friend went to trip around the world. He likes to take pictures of the places he visited and upload them online. We have given a template image of the world and we need to find where the picture has been taken.<br/> 
Pictures are patches extracted from template picture and can be various shape. They <ins>**will not be rotated**</ins>,  but they could have 2 kinds of distortions: 1) few random pixels value changes in patch 2) gaussian noise added to the patch. Input will be text files pointing to different sets of input (with names 1, 2, 3, 4…). There sets contains different patch size and patches with and without distortion. However same type of patches will be in same set. <br/><br/>
Our problem is to match each patch in template with 2 conditions: <br/>
	&emsp;1. Patch will be consider as matched if we are match left top corner in <ins>**40x40 area around expected point**</ins> <br/>
  &emsp;2. Patch matcher should <ins>**match patch in average time of 10ms**</ins> <br/><br/>
Template, sample input configuration and few different patch type images are shown below.
We can see that inputs folder contains [number].txt files referencing folder set/[number] which contains patches to be matched. First line in [number].txt file is path to map image. Second line (1000) is number  of patches in that folder and third line (40 40) is patch size. Then other lines represent path to each patch in that folder. <br/><br/>
![image](https://user-images.githubusercontent.com/24530942/210841435-987d33eb-230c-41ed-ba87-73875be6a7b8.png)

</p>

## Problem Solution

<p align="justify">

To solve this problem we implemented patch matcher. Patch matcher extract relevant points (key points) from template image and in these points it extracts feature which represent that point. These information in stored in patch matcher, so when we provide patch it goes through same procedure and extract relevant features from patch. Then it tries to find most suitable match from template features. When it does we find transformation which transforms patch location to template location.
<br/><br/>
<ins>**_Matching procedure can be summarized as follows:_**</ins><br/> 
	&emsp;1. &ensp; Extract relevant key points and feature around each key point from template image <br/>
	&emsp;2. &ensp; Save template key points and features in PatchMatcher object <br/>
	&emsp;3. &ensp; For each patch run same procedure: extract patch key points and features <br/>
	&emsp;4. &ensp; Match current patch features and saved template features <br/>
	&emsp;5. &ensp; Calculate transformation which transforms patch key points to template key points from matched features points <br/>
  
<ins>There are two patch matcher implementations. **Everything is implemented using just numpy and pandas library:**</ins> <br/>
	&emsp;1) &ensp; _Simple patch matcher:_ <br/>
		&emsp;&emsp;a. &ensp; key points - **each pixel** in template image; **center** of patch <br/>
		&emsp;&emsp;b. &ensp; features - **pixel values** around key point (20x20 area) <br/>
		&emsp;&emsp;c. &ensp; matching - minimum feature **Euclidian distance** <br/>
		&emsp;&emsp;d. &ensp; outlier filter - **no filter** <br/>
	&emsp;2) &ensp; _Advanced patch matcher:_ <br/>
		&emsp;&emsp;a. &ensp; key points - **corners** and high response value **edge pixels** got from **simplified SIFT algorithm** implementation <br/>
		&emsp;&emsp;b. &ensp; features - **gradient histogram** around key points (7x7 area) <br/>
		&emsp;&emsp;c. &ensp; matching - **distance between first and second nearest** feature <br/>
		&emsp;&emsp;d. &ensp; outlier filter - **RANSAC filter** <br/>

Simple patch matcher is slow and not robust to patch distortion so it will not be discussed in details. However, advanced patch matcher is robust to patch distortion and its fast enough to meet task conditions. Here will be presented in more details Advanced Patch Matcher and results got using this algorithm.
</p>

## Project architecture (in progress) <br/>

## Examples (in progress) <br/>

<p align="justify">

_Template Key Points_<br/>
![template_key_points](https://user-images.githubusercontent.com/24530942/210603843-ecd81551-477d-4aab-b592-0e4a6c0e6e16.png)<br/>

_Example of match patch_<br/>
![match](https://user-images.githubusercontent.com/24530942/210603888-75e7fa39-67e9-43d1-ab72-36e5f79109f2.png)<br/>
</p>

## Results (in progress) <br/>
