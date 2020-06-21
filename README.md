# Channel Estimation for One-Bit Multiuser Massive MIMO Using Conditional GAN
## 1. Description
This repository includes source codes of the paper "Channel Estimation for One-Bit Multiuser Massive MIMO Using Conditional GAN"

## 2. Run cGAN to Perform Channel Estimation
1. The dataset is already genreated ***"Data_Generation_matlab/Gan_Data/Gan_0_dBIndoor2p4_64ant_32users_8pilot.mat"***, which inculdes the channel data and quantized siganl data.
2. Run the main function ***"cGAN_python/main_cGAN.py"***. 

For each epoch, results will be save in the folder ***"cGAN_python/Results"*** and will show visual results as follows.

![image](https://github.com/YudiDong/Channel_Estimation_cGAN/blob/master/cGAN_python/generated_img/img_1.png)

## 3. How to Generate Data
1. Download "I1_2p4.zip" from this link: https://drive.google.com/drive/folders/1rbIHfK__JUn5e52y5GWI7p-0cL5OSZUO?usp=sharing. Then, extact "I1_2P4" folder and put it in the floder ***"Data_Generation_matlab/RayTracing Scenarios"***.
2. Run the matlab function ***"Data_Generation_matlab/GenerateData_Main.m"*** to generate channel data and quantized siganl data.

## 4. Referenced Repository

[1] https://github.com/Baichenjia/Pix2Pix-eager

[2] https://github.com/DeepMIMO/DeepMIMO-codes
