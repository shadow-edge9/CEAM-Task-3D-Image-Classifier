# CEAM 3D Classifier
**CEAM PROJECT:** A CNN ML Model built using PyTorch (TorchVision). It is a 3D Image Classifier trained to recognize three 3D shapes (cone, cube, sphere). Trained on a Dataset crafted from scratch with 300+ Images, this model is quite accurate in its predictions with an ~90% Training Accuracy and 75% Validation Accuracy. 

## Project Overview
The idea was to make a CNN that would differentiate between three easily recongnizable shapes : a cone, a cube and a sphere. 

In my earlier project on Fashion MNIST Dataset, I used TensorFlow and the Keras Library to make the Fashion Classifier. However, this project involved 3D figures of various colours, shapes, orientations and lighting conditions, in comparision to the Fashion MNIST Dataset which contained only 2D Images and greyscale. Hence I upgraded to PyTorch.

 ### STEP 1: DATASET CREATION
First I generated solid colour 3D models renderings in Blender with a script I generated using AI (because I'm not familiar with Blender or the bpy model) and ran the script within Blender to produce 60 randomly generated 3D rendering of cones, cubes and spheres, all in various colours, sizes, orientations and lighting conditions. 

My initial dataset consisted of 72 images (60 Training and 12 Validation). Later while running my model, it was still doing poorly even at the 3rd Trial, despite tweaking the epochs, adding an extra convolutional layer and even adding greyscale to prevent the model from cheating. 

Therefore, I updated my dataset to include 300+ images (60 Validation Images) which includes:
* Solid Colour 3D Rendered Figures (different sizes, colours, orientations, lighting conditions)
* Gradient Coloured 3D Rendered Figures (different sizes, colours, orientations, lighting conditions)
* 3D Figures against different coloured backgrounds (From the Internet)
* Real life object images (eg. dice, rubik cubes, traffic cones, party caps, marbles, balls, etc. from the Internet)

> The scripts I generated for Blender are in the repository. There are two separate scripts: 
> 1. shape_generator_1.py is the solid colour script. 
> 2. gradient_shape_generator.py is the gradient colour script.

### STEP 2: BUILDING THE MODEL
1. For all the modules used in the project, please refer to **Prerequisites**. First I started by creating a class called ```CEAMShapeNet``` that is basically a child class from the ```nn.Module``` class in ```torchvision```. Initially it had on 2 ```Conv2D``` layers but I later updated it to have 3 Conv2D layers due to the lack of precision my model was showing (rather, the absence of precision. It was consistently yelling CUBE! at every shape I threw it) Then I set up the Dense Layers are defined the forward function.
2. By default I set the use_dropout to False but later activate it during training so that the model doesn't get lazy and learn shortcuts.
3. After that I set up transforms in ```__main__``` for both Training and Validation Datasets. To prevent my model from cheating and learning only based on colour I stripped it off its ability to see colour using ```transforms.Grayscale```. Moreover I used ```transforms.RandomHorizontalFLip``` and ```transforms.RandomRotation``` to effectively increase the dataset images without physically doing any of these operations on my actual datatset.
4. Used ```os``` module to map my directories and load assets.
5. The optimizer I used was Adam. It's defaults work well for almost any Neural Network and also does well with noisy or inconsistent data, like the data we have here.
6. That said, the loops were made to parse images and train the model and then the model would give it's validation 'exam' (as I like to call it).
7. The accuracy and loss of both Training and Validation was then collect and stored in a list defined at the beginning of the ```__main__``` to be able to produce line graphs of the same. The console also prints the raw data in real time as each Epoch completes.
8. At last the weights would be saved in a .pth file for the ```predict.py``` script.

It took me a total of 6 Trials to get to the near accurate version of this model.

### STEP 3: THE REAL DEAL - THE PREDICTOR
Now that I had my model, the brain of my project all prepared and ready for action, I had to use it. Somewhere. 

`predict.py` script is the main script to be run since I've already uploaded the weights from the last run of `model.py`. Here I made a new function `predict_custom_image` which contains all the instructions to predict any image that is uploaded from any directory (the ones I used for testing is available in `Test` folder. 
> Make sure you write the file paths properly while testing or else you'll get an error.

### STEP 4: TEST THE MODEL ON EVERY CASE
I tried my level best to see where this model fails by testing it with nearly 30+ images of different parameters all based on my training data. This was genuinely the fun part because I wanted to prove my initial hypothesis right, but turns out the model was smarter than I accounted it for!


## Metrics and Parameters
* **Optimizer** : Adam
* **Learning Rate** : 0.001
* **Batch Size** : 32
* **Regularization** : Dropout (p=0.5) enabled during training.
* **Convolutional Layers** : 3
* **Colour (RGB)** : Currently disabled
* **Batch Shuffle** : True

## Prerequisites
This project was built with Python 3.

Modules used in this project are:
* `os`
* `torch`
* `seaborn` and `matplotlib` for graphs
* `PIL` (for `Image`)

## How to Run the Project
1. You do NOT need to run `model.py`. The weights have been uploaded already. Do not attempt to delete the weights file or else you will not be able to run `predict.py`. (Training Dataset cannot be uploaded due to limitations)
2. Run `predict.py` after typing in the image path. You have at your disposal a collection of test images I used myself in the `Test` directory.

> If you have Blender, you can run `shape_generator_1.py` and `gradient_shape_generator` **within Blender** NOT in Python. The script will not run there. The script will generate beautiful 3D renders for you and save it to a folder.

## Experimentation Phase
## Observations and Conclusions
This section is under construction...
## Learnings and Challenge and Bug Fixes
This section is under construction...
## Next Step
* I intend to run the model without the greyscale filter and see if it's accuracy improves or worsens
