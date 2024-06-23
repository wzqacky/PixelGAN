# Summary 

This repository serves as an exercise to implement Generative Adverserial Network (GAN), specifically the Wasserstein GAN to generate images. 

# Dataset
Curated Pixel Art 512x512 
Available [here](https://www.kaggle.com/datasets/artvandaley/curated-pixel-art-512x512)

# Structure of repo 
<b>Dataset</b>
- Contains the pixel art images for training the model

<b>pretrained_model</b>
- Directory for saving models
- Contains a pair of pretrained generator and discrminator (25,000 epochs with learning rate=1e-4)

<b>train.py</b>
```
python train.py --model LSTM --num_epochs 500
```
#### Optional parameters:
- num. of epochs, default=500
- learning rate, default=1e-4
- gradient penalty, default=10
- discriminator iterations, default=5
- save model, default=True

<b>generate_image.py</b>
```
python generate_image.py --model generator25000.py
```
#### Optional parameters:
- model, default=generator25000.py

Samples images:
![sample](https://github.com/wzqacky/PixelGAN/assets/100191968/ef79fcdb-9730-45f2-b516-be4da851be93)

