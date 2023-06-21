# Normal Map Generator

## Methodology
* Creation of a working model which uses a Unet architecture on PyTorch with PyTorch Lightning and plot results on Tensorboard.
  * The model is a simple Unet architecture, with encoder / decoder. The decoder uses upsampling and convolutional layers to avoid the checkerboard effect.
  * Data augmentation is used (random flip, random crop)
* Tested different models parameters
  * Initial channel is 64, tested with 32 and 128
* Added a discriminator
* Added data normalization (-1, 1)

## Results

All models except the larger model were trained for 100 epochs.

| Iteration   | Train loss  | Val Loss   | Generated Results   |   
|---|---|---|---|
| Ground Truth  |     |     |  <img src="images/ground_truth.png"  width="200" height="200">  |   
| Initial Model  | ![](images/0_initial_model/train_rec_loss.png)   |  ![](images/0_initial_model/val_rec_loss.png)  |  <img src="images/0_initial_model/generated.png"  width="200" height="200">  |   
| Smaller model (initial channel at 32)  | ![](images/1_smaller_model/train_rec_loss.png)   |  ![](images/1_smaller_model/val_rec_loss.png)  |  <img src="images/1_smaller_model/generated.png"  width="200" height="200">  |   
| Larger model (initial channel at 64)  | ![](images/2_larger_model/train_rec_loss.png)   |  ![](images/2_larger_model/val_rec_loss.png)  |  <img src="images/2_larger_model/generated.png"  width="200" height="200">  |   
| Add Discriminator  | ![](images/3_discriminator/train_rec_loss.png)   |  ![](images/3_discriminator/val_rec_loss.png)  |  <img src="images/3_discriminator/generated.png"  width="200" height="200">  |   
| Normalize data between -1 and 1  | ![](images/4_normalization/train_rec_loss.png)   |  ![](images/4_normalization/val_rec_loss.png)  |  <img src="images/4_normalization/generated.png"  width="200" height="200">  |   