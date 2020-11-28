# Channel Estimation for One-Bit Multiuser Massive MIMO Using Conditional GAN
## 1. Description
This repository is the implenation of the paper: 
Yudi Dong, Huaxia Wang, and Yu-Dong Yao, “Channel Estimation for One-Bit Multiuser Massive MIMO Using Conditional GAN.” ArXiv:2006.11435 [Eess], June 2020. arXiv.org, http://arxiv.org/abs/2006.11435.
The paper is accepted in IEEE Communications Letters, DOI: 10.1109/LCOMM.2020.3035326


## 2. Run cGAN to Perform Channel Estimation (TensorFlow Version is 2.0)
1. The dataset is already genreated ***"Data_Generation_matlab/Gan_Data/Gan_0_dBIndoor2p4_64ant_32users_8pilot.mat"***, which inculdes the channel data and quantized signal data.
2. Run the main function ***"cGAN_python/main_cGAN.py"***. 

For each epoch, results will be saved in the folder ***"cGAN_python/Results"*** and will show visual results as follows.

![image](https://github.com/YudiDong/Channel_Estimation_cGAN/blob/master/cGAN_python/generated_img/img_1.png)

## 3. How to Generate Data
1. Download "I1_2p4.zip" from this link: https://drive.google.com/drive/folders/1rbIHfK__JUn5e52y5GWI7p-0cL5OSZUO?usp=sharing. Then, you should extact "I1_2P4" folder and put it in the folder ***"Data_Generation_matlab/RayTracing Scenarios"***.
2. Run the matlab function ***"Data_Generation_matlab/GenerateData_Main.m"*** to generate channel data and quantized signal data.

## 4. Referenced Repository

[1] https://github.com/Baichenjia/Pix2Pix-eager

[2] https://github.com/DeepMIMO/DeepMIMO-codes
