# <p align="center"> AUTONOMOUS NAVIGATION ON ROAD </p>

<p align="center">
  [Watch the video](https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/short_highquality_video.gif)
</p>
  
Personal cars and commercial trucks are continuously improving the driver experience and safety thanks to integration of more significant and machine-assisted control systems. Advanced driver-assistance systems (ADAS) are now integrated in all luxury cars and moving into mainstream products. Technologies covered by ADAS are specific for each car integrator, but increasingly they include now involving more safety features, such as driver assistance and partial delegation to autonomous control for small maneuvers such as lane control. The ADAS systems consists of control system such as:

- Lane departure warning system
- Speed assistance and control
- Autonomous emergency braking

## Projects tasks
### 1) Autonomous lane keeping: 
Given an on-track camera video, detect the left and right lane markers (Python program / 1st). 

#### Pipeline for lane detection
##### 1.1 Camera calibration:
To compute the camera the transformation matrix and distortion coefficients, we use multiple pictures of a chessboard on a flat surface taken by the same camera. OpenCV has a convenient method called FindChessboardCorners that will identify the points where black and white squares intersect and reverse engineer the distortion matrix this way. The image below shows the identified chessboard corners traced on a sample image and camera parameters computed from it.

<p align="center">
  <img width="400" height="150"
       src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/camera_calibration.png">
</p>

##### 1.2 Distortion removal: 
Calibration matrix applied to a distorted image (left) produces undistorted image (right)
<p align="center">
  <img width="350" height="130" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/calibrated%20image.jpg">
</p>

##### 1.3 Thresholding: 
One of the most challenging part of lane detection is to use appropriate methods for converting RGB image to binary image that only consist of lane lines. For this several methods were explored including: 
- RGB image - > Grayscale image -> magnitude-based thresholding
- RGB -> Grayscale -> directional sobel filter
- RGB –> HLS -> Sobel on L channel -> Threshold on all channels 

The problem here is there are a lot of parameters that can be tuned; min and max thresholds, different color spaces, x and y direction and so on. Parameter space is very large, and it is very hard to hit a good solution just by the method of trial and error. After comparative study through research papers, solution based on HLS / HSV seemed to be perfect for colored lane lines. By using filters on colors and then on lightness parameters, it can robustly identify the blue lane lines from image. (This method can be used for any other as well, or when finding out lane from multiple colored lane lines). The only problem is that it is not very robust with strong shadows, where it can have a lot of intensity in some areas.

<p align="center">
  <img width="400" height="150" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/thresholding.jpg">
</p>

##### 1.4 Region of interest and perspective transform (Birds eye view):
After getting binary image in the previous step, perspective transform was applied to it to change it from camera view into top-down view.

<p align="center">
  <img width="250" height="150" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/ROI.jpg">
</p>

<p align="center">
  Figure: Region of interest
</p>

<p align="center">
  <img width="400" height="150" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/birdseyeview.jpg">
</p>
<p align="center">
  Figure: Perspective transform (left) vs original image (right)
</p>

##### 1.5 Sliding window methods to detect lane line pixels: 
This part of the pipeline was also very challenging to implement in a robust way. The goal here is to create a mask for left and right road lines to only keep pixels of the lines and not anything else. First, we get initial positions of lanes by using half of the image to calculate histogram and detect 2 peaks. Then, we split input image into horizontal strips. After that, for each strip, we try to detect two peaks, this is where centers of lanes are. Then we also create two empty (zero-valued) masks for left and right lane. For each peak we will take a predefined no. of pixel window to each side of each peak and make this window one-valued in the mask. After we did this, we will have two masks that we can apply to the binary image. After we have 2 fitted lines, we can simplify the search for masks. We can argue that lines in two consecutive frames will be close to each other. Therefore, the mask can be calculated as windows sitting on the fitted polynomial, which really speed up calculations.

<p align="center">
  <img width="400" height="150" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/slidingwindow.jpg">
</p>
<p align="center">
  Figure: Histogram analysis (left image) to find two distinct lane lines
</p>

##### 1.6 Fitting quadratic lines to the masked pixels: 
It takes a masked binary image and calculates coordinates for all non-zero points. Then these points are used to fit a second-order polynomial using np.polyfit() function. The part was to remember that horizontal position is dependent variable and vertical is independent. Therefor the fitting a function:
<p align="center">
  x = f(y) = a * y**2 + b * y + c
</p>

<p align="center">
  <img width="400" height="300" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/slidingwindow2.jpg">
</p>
<p align="center">
  Figure: Input image(top), perspective view(bottom-right) and final lane pixels with sliding box(bottom-left)
</p>

##### 1.7 Computation of lane center, vehicle center and visualize lane: 
It From step 6, we received the parameters of quadratic equation of two-lane lines. In image co-ordinate frame (u, v), by using v = 0 and v = fixed_look_ahead_point, the (u, v) coordinates of upper and lower lane line point can be calculated for both lane lines. Median of upper lane points will give upper lane center (highlighted by white circle) and from lower lane point lower lane center point (highlighted in green circle). Since camera is mounted on vehicle and has fixed orientation (assumption) the camera center and vehicle center can be assumed to be same. Vehicle center is highlighted in color red. By taking inverse perspective view of identified lane line pixels from step 4 and using weighted image, lane lines with blue and red color ; and original image can be blended to visualize output results.

<p align="center">
  <img width="400" height="250" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/Picture1.jpg">
</p>
<p align="center">
  Figure: Output from lane detection pipeline
</p>

### 2) Road Sign Recognition:
Given an on-track camera video, recognize the stop sign and school zone sign in the video and mark them using bounding boxes (MATLAB program / 2nd). 

Approach used: Machine learning using Convolutional Neural Network (CNN) 

Advantages of using CNN:
- High accuracy and performance in terms of detection time
- Option of transfer learning
- Requires training only on one neural network for multiple sign detection class.

A machine learning based algorithm was used to determine road signs from the video frames. This algorithm was implemented in MATLAB using its default cascade object detector. To create a detector object file, we need to first train the algorithm with a bunch of positive and negative images. In this project, we are supposed to detect two distinct road signs, a STOP sign and a School sign.

<p align="center">
  <img width="500" height="150" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/signdetection.jpg">
</p>
<p align="center">
  Figure: Sign detector process flow
</p>

<p align="center">
  <img width="250" height="150" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/detectedsign.jpg">
</p>
<p align="center">
  Figure: Sign detector example
</p> 
 
The sign detector gives the output in form of which sign is detected in given frame and predicted accuracy of sign. This information is then communicated to PC running lane detector. ‘STOP’, ‘SCHOOL’ or ‘Continue’ messages were sent in string/character format.

### 3) Communications: 
Send the road sign information from the 2nd program to the 1st program through communications. The two programs can be on either the same computer or two different computers.

Establishing communication between two different computers running was very crucial in the project as lane detection and control and road sign recognition were carried out on two different computers. A User Datagram Protocol (UDP) was implemented for communication. The flow of server-client kind of architecture can be seen in the figure below.
<p align="center">
  <img width="250" height="150" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/UDP.jpg">
</p>
<p align="center">
  Figure: Server-client type UDP based communication
</p> 

Since both the computers were using individual video feed for lane detection and sign recognition, the syncing between them was one of the major issues in communication. To overcome this issue, we decided to establish a two-way communication where a lane detection computer send a frame indexing signal to the sign recognizing computer. Sign recognizing computer will first check the video frame, store the results and will wait for the indexing signal. After receiving indexing signal, it will send the result of sign recognition to the lane detecting computer. Lane detecting computer will only proceed further after receiving the data from the second computer. This method solved the issue of video syncing but at the same time, it slows down the whole process as both the computer now need to wait for the confirmation from the other computer.

### 4) Vehicle Controls and HMI: 
From detected lane, calculate steering angle uding Stanley Control. Design appropriate HMI to intuitively display control commands (e.g., STOP, SLOW DOWN) in the image at stop sign and school zone sign in the 1st program.

##### 4.1 Inverse Projection 
To design Stanley control, first camera frame co-ordinates need to be calculated from pixel frame co-ordinates.

<p align="center">
  <img width="450" height="150" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/pixeltocamera.jpg">
</p>
<p align="center">
  Figure: Pixel frame to camera frame transformation
</p> 

From last step of autonomous lane detection, pixel co-ordinates of vehicle center, two points of lane center were derived which will be used for calculating image frame coordinates of same.
<p align="center">
  <img width="200" height="80" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/pixeltocamera2.jpg">
</p>
<p align="center">
  Figure: Pixel frame to camera frame transformation
</p> 

Since, Fu, Fv and u0, v0 where derived from camera calibration step. Yc is assumed to be 210mm (assuming road surface is flat, and camera is mounted at the center of vehicle). From all the steps above, we now have local frame coordinates (camera frame) which can be used for calculating steering angle through Stanley control.

<p align="center">
  <img width="400" height="300" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/vehicledepartureangle.jpg">
</p>
<p align="center">
  Figure: Derivation of theta_e (Vehicle departure angle)
</p> 
All the steps to calculate lane center and vehicle heading were already calculated in lane detection algorithm and thus simple pixel to camera frame transformation is used to get inputs for Stanley control.

##### 4.2 Stanley control for steering
Vehicle motion model can be defined as:
<p align="center">
  <img width="300" height="100" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/stanleycontrol.jpg">
</p>
Here, x, y, 𝜃 are vehicles pose while 𝜑 is steering angle, V is vehicle velocity. Then the Stanley method is used to find the steering angle 𝛿 to drive the vehicle (front axle). The closest point of the desired path (efa), it will be perpendicular distance from vehicle front axle point to line. Along with the tangent of the desired path at the closest point: 
<p align="center">
  <img width="300" height="300" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/stanleycontrol2.jpg">
</p>
Thus, steering angle is defined as:
<p align="center">
  <img width="700" height="355" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/steeringangle.png">
</p>

##### 4.3 HMI
HMI to intuitively display control commands (e.g., STOP, SLOW DOWN) in the image at stop sign and school zone sign in the 1st program.

<p align="center">
  <img width="750" height="450" src="https://github.com/vipulkumbhar/AuE824_Autonomous_Driving_Technologies/blob/master/AuE8240_Team8/Presentation/17%20process%20flow%20.jpg">
</p>
<p align="center">
  Figure: Process flow-based display (easy to debug)
</p> 

###### Displayed information:
1) Original frame with detected lanes embedded in it. Also, left and right lanes are colored differently so that user can see /check that detected lane lines are correct.
2) Steering angle: Steering angle derived from Stanley control is displayed. For easier interpretation angle is displayed in degrees.
3) Lane error: This is cross-track error (configurable, either in pixel coordinates or camera frame coordinates). This information can be used for determining effects of cross-track error on steering angle determined by Stanley control.
4) FPS: frames per second processed by system. This information can be used for determining performance of system.
5) Sign detected: Whenever sign detected information is conveyed by remote PC, it is displayed on screen (either ‘Stop sign detected’ or ‘school’ sign detected). As per sign detected big message is displayed in center of frame (Either ‘STOP’ or ‘Reduce speed’)
6) Lane departure warning: If cross-track error is higher than pre-defined values, lane departure warning is displayed on screen.
7) Vehicle center (red color circle), lane center at vehicle (green color circle) , lane center at some fixed distance from vehicle (white circle)
