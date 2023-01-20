# Patch match algorithm

## Table of contents
1. [Problem statement](#p1)
2. [Problem solution description](#p2)
3. [Project architecture](#p3)
4. [Patch match examples](#p4)
5. [Patch match results](#p5)

## Problem statement <a name="p1"></a>
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

## Problem solution description <a name="p2"></a>

<p align="justify">

To solve this problem we implemented patch matcher. Patch matcher extract relevant points (key points) from template image and in these points it extracts feature which represent that point. These information in stored in patch matcher, so when we provide patch it goes through same procedure and extract relevant features from patch. Then it tries to find most suitable match from template features. When it does we find transformation which transforms patch location to template location.
<br/><br/>
<ins>**_Matching procedure can be summarized as follows:_**</ins><br/> 
	&emsp;1. &ensp; Extract relevant key points and feature around each key point from template image <br/>
	&emsp;2. &ensp; Save template key points and features in PatchMatcher object <br/>
	&emsp;3. &ensp; For each patch run same procedure: extract patch key points and features <br/>
	&emsp;4. &ensp; Match current patch features and saved template features <br/>
	&emsp;5. &ensp; Calculate transformation which transforms patch key points to template key points from matched features points <br/>
  
<ins>There are two patch matcher implementations. **Everything is implemented using only numpy:**</ins> <br/>
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

## Project architecture <a name="p3"></a>
<p align="justify">
	
Project structure can be found in picture below. Modules _kpi_calculation_ and _patch_matcher_ are core modules which are fully implemented using **only numpy**. Module _patch_matcher_visualisation_ serves or visualization purposes and it **has opencv dependency**. Folder _reports_ its generated using KPI module and has KPI reports and plots. Script _main_patch_match.py_ is main function and _config folder_ has yaml configuration file.<br/><br/>	
	<img src="https://user-images.githubusercontent.com/24530942/213715365-b393fcc1-8eb2-4150-b28e-8ab7a481a0ac.PNG" height="350" width="250">
	<br/><br/>
	
<ins>**_We can see below pictures of UML diagrams for two major modules of Patch Match algorithm._**</ins><br/><br/>
	&emsp;1. &ensp; **UML Class Diagram for Patch Match** class has base class PatchMatcher which serves as interface and two class implementations SimplePatchMatcher and AdvancedPatchMatcher. <br/><br/>
	<img src="https://user-images.githubusercontent.com/24530942/213717140-45006ef9-8ffc-45e7-8482-da27f3fb70ba.png" height="450" width="700">
<br/><br/>
	&emsp;2. &ensp; **UML Class Diagram for generating KPI results** has class CalculateKPI which serves for KPI calculation. It is using PatchMatcher class for matching input patches and Report class to generate (save) reports for each input.
 <br/>
	<img src="https://user-images.githubusercontent.com/24530942/213721557-34314428-77ce-4296-93e5-c13d5ee10dd5.png" height="350" width="700">
</p>

## Patch match examples <a name="p4"></a>

<p align="justify">

For generating example _AdvancedPatchMatcher_ is used. First image is showing <ins>template image key points</ins> which are got using **simplified SIFT algorithm** implementation. Second image is showing example of <ins>successfully matched patches</ins> (with its key points) and <ins>matched key points</ins> on template image.
<br/><br/>
	**<ins>_Template key points_</ins>**<br/>
![image](https://user-images.githubusercontent.com/24530942/213725464-4b04e607-882e-43e0-ad8f-c43ce08ff37b.png)
<br/>
**<ins>_Example of successfully matched patches_</ins>**<br/>
![image](https://user-images.githubusercontent.com/24530942/213725690-84daad2c-5eef-481b-bbe7-c90e9d9f0be8.png)<br/>
</p>

## Patch match results (in progress) <a name="p5"></a>
