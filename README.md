# AuE8240_Autonomous_Driving_Technologies


Personal cars and commercial trucks are continuously improving the driver experience and safety thanks to integration of more significant and machine-assisted control systems. Advanced driver-assistance systems (ADAS) are now integrated in all luxury cars and moving into mainstream products. Technologies covered by ADAS are specific for each car integrator, but increasingly they include now involving more safety features, such as driver assistance and partial delegation to autonomous control for small maneuvers such as lane control. The ADAS systems consists of control system such as:

- Lane departure warning system
- Speed assistance and control
- Autonomous emergency braking

## Projects tasks
### 1) Autonomous lane keeping: 
Given an on-track camera video, calculate the steering angle of the vehicle to track the lane, and design appropriate HMI to intuitively display the steering control in the image (Python program / 1st). 

<p align="center">
  <img width="500" height="300" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/Picture1.jpg">
</p>

#### Pipeline for lane detection
##### 1.1 Camera calibration:
To compute the camera the transformation matrix and distortion coefficients, we use multiple pictures of a chessboard on a flat surface taken by the same camera. OpenCV has a convenient method called FindChessboardCorners that will identify the points where black and white squares intersect and reverse engineer the distortion matrix this way. The image below shows the identified chessboard corners traced on a sample image and camera parameters computed from it.

<p align="center">
  <img width="500" height="300" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/camera_calibration.png">
</p>

##### 1.2 Distortion removal: 
Calibration matrix applied to a distorted image (left) produces undistorted image (right)
<p align="center">
  <img width="500" height="300" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/calibrated%20image.jpg">
</p>

##### 1.3 Thresholding: 
One of the most challenging part of lane detection is to use appropriate methods for converting RGB image to binary image that only consist of lane lines. For this several methods were explored including: 
- RGB image - > Grayscale image -> magnitude-based thresholding
- RGB -> Grayscale -> directional sobel filter
- RGB –> HLS -> Sobel on L channel -> Threshold on all channels 

The problem here is there are a lot of parameters that can be tuned; min and max thresholds, different color spaces, x and y direction and so on. Parameter space is very large, and it is very hard to hit a good solution just by the method of trial and error. After comparative study through research papers, solution based on HLS / HSV seemed to be perfect for colored lane lines. By using filters on colors and then on lightness parameters, it can robustly identify the blue lane lines from image. (This method can be used for any other as well, or when finding out lane from multiple colored lane lines). The only problem is that it is not very robust with strong shadows, where it can have a lot of intensity in some areas.

<p align="center">
  <img src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/thresholding.jpg">
</p>

##### 1.4 Region of interest and perspective transform (Birds eye view):
After getting binary image in the previous step, perspective transform was applied to it to change it from camera view into top-down view.

<p align="center">
  <img width="250" height="150" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/ROI.jpg">
</p>

<p align="center">
  <img width="400" height="150" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/birdseyeview.jpg">
</p>



##### 1.2 Distortion removal: 
##### 1.2 Distortion removal: 
##### 1.2 Distortion removal: 

### 2) Road Sign Recognition:
Given an on-track camera video, recognize the stop sign and school zone sign in the video and mark them using bounding boxes (MATLAB program / 2nd).

### 3) Communications: 
Send the road sign information from the 2nd program to the 1st program through communications. The two programs can be on either the same computer or two different computers.

### 4) Vehicle Controls: 
Design appropriate HMI to intuitively display control commands (e.g., STOP, SLOW DOWN) in the image at stop sign and school zone sign in the 1st program.


