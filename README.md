# AlColor	

AIColor is our code for NTIRE 天池大数据比赛AI着色 CVPR2023

![image-20230926102211044](https://github.com/aoxipo/AIColor/blob/main/assert/banner.png)

# Model

we propose the Mult DDIM on HourGlass model to Enhance image color

Build different Unet models for different time steps to accelerate the model inference and training process.

# Train

#TODO

# Results

### CDC Score

![image-20230926101335076](https://github.com/aoxipo/AIColor/blob/main/assert/cdc_first.png)

![image-20230926101430415](https://github.com/aoxipo/AIColor/blob/main/assert/cdc_second.png)



### FID Score

![image-20230926101605007](https://github.com/aoxipo/AIColor/blob/main/assert/fid_first.png)

![image-20230926101626267](https://github.com/aoxipo/AIColor/blob/main/assert/fid_second.png)

Here is our result in tianchi rank board result and we got 11/93 adn 10/130 rank in FID and CDC

we test different model to generate image after colored

###### DF-gan tiny result:

![10](./assert/save/dfgan_tiny/10.png)

###### DF-gan middle result:

![30](./assert/save/dgan_64_128/30.png)

###### DF-Unet-Gan result

![65](./assert/save/PHDAE/65.png)

###### DF-HourGlass-Gan result

![40](./assert/save/dqgan_origin/40.png)

![30](./assert/save/dqgan_origin/30.png)

###### 
