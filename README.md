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
  <img width="500" height="300" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/camera_calibration.png.jpg">
</p>

### 2) Road Sign Recognition:
Given an on-track camera video, recognize the stop sign and school zone sign in the video and mark them using bounding boxes (MATLAB program / 2nd).

### 3) Communications: 
Send the road sign information from the 2nd program to the 1st program through communications. The two programs can be on either the same computer or two different computers.

### 4) Vehicle Controls: 
Design appropriate HMI to intuitively display control commands (e.g., STOP, SLOW DOWN) in the image at stop sign and school zone sign in the 1st program.


