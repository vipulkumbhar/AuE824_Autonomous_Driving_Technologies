#!/usr/bin/env python
import socket
from time import sleep
import time
import yaml
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import math

# UDP / connect to other system

# Local host
UDP_IP    = "192.168.1.114"
UDP_PORT  = 5006

# Sign recognition / remote PC
UDP_IP2   = '192.168.1.65'
UDP_PORT2 = 4004

sock = socket.socket(socket.AF_INET,    # Internet
                     socket.SOCK_DGRAM) # UDP

sock.bind((UDP_IP, UDP_PORT))

# Load camera calibration

#cameraCalibration = pickle.load( open
#                    ('Camera_Calibration_Images/distortion_coefficients/camera_calibration.p', 'rb' ) )
#mtx, dist = map(cameraCalibration.get, ('mtx', 'dist'))

# undistort image
def undistort_image(frame,mtx,dist):
    #undistort image
    undist = cv2.undistort(frame, mtx, dist, None, mtx)
    return undist

# crop area of interest
def region_of_interest(img):
    
    xSize, ySize= img.shape[0],img.shape[1]
    roi_left_top     = [ySize/4,xSize/2]
    roi_right_top    = [ySize*3/4,xSize/2]
    roi_right_bottom = [ySize-100,xSize]
    roi_left_bottom  = [100,xSize]

    vertices = np.array([[roi_left_top, roi_right_top, roi_right_bottom, roi_left_bottom]], dtype=np.int32)
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# RGB or gray
def birdseye_view(img,inverse=0,look_ahead=5):
    
    xSize, ySize= img.shape[0],img.shape[1]
    
    lower_offset = 10
    offset       = 100
    
    # define regiom of interest
    bottomY      = int(xSize)
    topY         = int(xSize/look_ahead)
    
    left1  = (lower_offset, int(bottomY))
    left1_x, left1_y = left1

    left2  = (int(ySize/2 - ySize/4), topY)
    left2_x, left2_y = left2

    right1 = (int(ySize/2 + ySize/4), topY)
    right1_x, right1_y = right1

    right2 = (ySize-lower_offset, bottomY)
    right2_x, right2_y = right2
    
    src = np.float32([ 
        [left2_x, left2_y],
        [right1_x, right1_y],
        [right2_x, right2_y],
        [left1_x, left1_y]
        ])
    
    img_size = (ySize, xSize)
    
    dst = np.float32([
        [offset, 0],
        [img_size[0]-offset, 0],
        [img_size[0]-offset, img_size[1]], 
        [offset, img_size[1]]
        ])
    
    M      = cv2.getPerspectiveTransform(src, dst)
    Minv   = cv2.getPerspectiveTransform(dst, src)
    
    if inverse ==0:
        img2 = cv2.warpPerspective(img, M, img_size)
    if inverse ==1:
        img2 = cv2.warpPerspective(img, Minv, img_size)
    
    return img2

def apply_gradient_and_thresholding(input_image, s_thresh=(100, 255), l_thresh=(80,255), sx_thresh=(20, 100)):
    
    # Convert to HLS color space and separate the V channel
    hls       = cv2.cvtColor(input_image, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Apply Sobel x
    sobelx       = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # This will take the derivative in x
    abs_sobelx   = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from the horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
      
    # Apply Thresholding 
    final_binary = np.zeros_like(s_channel)
    final_binary[np.logical_or((s_channel > s_thresh[0])
                               & (s_channel < s_thresh[1])
                               & (l_channel > l_thresh[0])
                               & (l_channel < l_thresh[1]),
                               (scaled_sobel > sx_thresh[0])
                               & (scaled_sobel <= sx_thresh[1]))] = 1
        
    return final_binary

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram   = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img     = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint    = np.int(histogram.shape[0]//2)
    leftx_base  = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin   = 100
    # Set minimum number of pixels found to recenter window
    minpix   = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero  = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current  = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds  = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low       = binary_warped.shape[0] - (window+1)*window_height
        win_y_high      = binary_warped.shape[0] - window*window_height
        win_xleft_low   = leftx_current - margin
        win_xleft_high  = leftx_current + margin
        win_xright_low  = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current  = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds  = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx  = nonzerox[left_lane_inds]
    lefty  = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped,frame):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    left_fit  = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx  = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        #center_fitx= (left_fit[0]/2+right_fit[0]/2)*ploty**2 + 
        #(left_fit[1]/2+right_fit[1]/2)*ploty + (left_fit[2]/2+right_fit[2]/2)
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx  = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
        #center_fitx= 1*ploty**2 + 1*ploty

    # Plot the polynomial lines onto the image
    left_pts  = np.array([np.dstack((left_fitx,ploty))], np.int32)
    left_pts  = left_pts.reshape((-1,1,2))
    right_pts = np.array([np.dstack((right_fitx,ploty))], np.int32)
    right_pts = right_pts.reshape((-1,1,2))
    
    #center_pts = np.array([np.dstack((center_fitx,ploty))], np.int32)
    #center_pts = right_pts.reshape((-1,1,2))
    
    cv2.polylines(out_img,[left_pts] ,False,(0,255,255),15)
    cv2.polylines(out_img,[right_pts],False,(0,255,255),15)
    #cv2.polylines(out_img,[center_pts],False,(255,0,255),15)

    Lane_pts = np.zeros_like(frame)#out_img)
    Lane_pts[lefty,leftx] = (0,0,255)
    Lane_pts[righty,rightx] = (255,0,0)
    #Lane_pts = cv2.polylines(Lane_pts,[center_pts],False,(0,255,255),5)

    return out_img,left_pts,right_pts,left_fit, right_fit, Lane_pts

def frame_test(img2):

    # Select region of interest and take birds eyes view
    #img2    = region_of_interest(img2)
    img2     = birdseye_view(img2,inverse = 0,look_ahead=2)
    img3     = img2

    # apply gradient threshold and optimize
    img2     = apply_gradient_and_thresholding(img2, s_thresh=(80, 255), l_thresh=(80,255), sx_thresh=(20, 100))
    kernel   = np.ones((4,4), np.uint8)
    img2     = cv2.erode(img2, kernel, iterations=1) 
    img2     = cv2.dilate(img2, kernel, iterations=3)

    # sliding box and fit polynomial 
    img2,left_pts,right_pts,left_fit, right_fit, Lane_pts = fit_polynomial(img2,frame)
    
    # take inverse birds eys view, images to superimpose on original image
    img2     = birdseye_view(img2,inverse = 1,look_ahead=2)
    Lane_img = birdseye_view(Lane_pts,inverse = 1,look_ahead=2)
    
    b = 0    #plt.imshow(img2, cmap='gray')

    return b,img2,Lane_img,img3,left_fit,right_fit

def groudn_cord(u,v):

	# camera calibration parameters
	kuf = 722
	kvf = 722
	u0  = 653 
	v0  = 492
	Yc  = 210
	# image frame to local co-ordinate system transoformation
	zc  = kvf*Yc/(v-v0)
	xc  = zc*(u-u0)/kuf
	return xc,zc

def stanley_control(img,left_fit,right_fit,look_ahead_distance = 500):
    
    vimg, himg            = img.shape[0],img.shape[1]
    lad                   = vimg - look_ahead_distance
    control = 1

    # co-ordinates of look ahead point
    left_lane_point       = left_fit[0]*lad**2  + left_fit[1]*lad  + left_fit[2]
    right_lane_point      = right_fit[0]*lad**2 + right_fit[1]*lad + right_fit[2]
    
    lad_point_h           = int( (left_lane_point + right_lane_point)/2)
    lad_point_v           = int( lad )
    lad_point             = (lad_point_h,lad_point_v)
    
    # lane center point at camera
    left_c_point          = left_fit[0]*vimg**2  + left_fit[1]*vimg  + left_fit[2]
    right_c_point         = right_fit[0]*vimg**2 + right_fit[1]*vimg + right_fit[2]
    lane_center_h         = int((left_c_point + right_c_point)/2)
    lane_center           = (lane_center_h,vimg)
    
    #vehicle center point 
    vehicle_v = vimg
    vehicle_h = (himg/2)

    # left_lane,right lane and vehicle ground plane cooridnates
    lux,luz   = groudn_cord(left_lane_point,lad)
    rux,ruz   = groudn_cord(right_lane_point,lad)
    llx,llz   = groudn_cord(left_c_point,vimg)
    rlx,rlz   = groudn_cord(right_c_point,vimg)
    vcgx,vcgz = groudn_cord(vehicle_h,vehicle_v)
	
    # center point pixel co-oridnates
    ladgx = int((lux + rux)/2)
    ladgz = int((luz + ruz)/2)
    ladg  = (ladgx,ladgz)

    lcx   = int((llx + rlx)/2)
    lcz   = int((llz + rlz)/2)
    lcg   = (lcx,lcz)
    
    # theta-e and ef based Staley control
    if control ==1:
    	check = lad_point_v-vimg
    	if check==0:
        	theta_e = 1.57 
    	if check != 0:
        	theta_e = math.atan2((lad_point[0] - lane_center[0]),(-lad_point[1]+lane_center[1]))
    	ef      = float((himg/2 - lane_center_h)*math.cos(theta_e))

    if control ==0:
	theta_e = math.atan(float(ladgz-lcz)/float(-ladgx+lcx)) + 1.16937
    	ef      = float((vcgx - lcx)*math.cos(theta_e))
    
    # steering angle from stanley
    phi = -theta_e + ef/400
        
    #theta e in degree
    theta_e = theta_e*180/3.14159
    
    return lad_point,lane_center,theta_e, ef, phi,ladg,lcg


# Initiate video capture and perform lane detection / Stanley control / Sign recognition from remote pc and HMI

cap = cv2.VideoCapture('video/Clockwise.mp4')                  # Bot video
#cap = cv2.VideoCapture('video/Real_vehicle_video.mp4')        # Real video / need to change lot of parameters for conclusive lane detection

n       = 1 # initiate frame no
t       = 0 # initiate time 
connect = 0 # Connect to remote PC (0 -NO, 1 -Yes)
while(True):
    ret, frame = cap.read()
    
    if ret:
        
        # Lane detector
        b, img2,Lane_img,img3,left_fit,right_fit = frame_test(frame)
        
        # stanley controller
        lad_point,lane_center,theta_e, ef,phi,ladg,lcg = stanley_control(frame,
                                                                    left_fit,right_fit,
                                                                    look_ahead_distance = 400)
        

        # FPS
        t0        = time.time()
        time_diff = t0-t
        t         = t0
        
        # print / display
        vimg, himg          = frame.shape[0],frame.shape[1]
        
        combined_frame = cv2.addWeighted(frame, 0.5, Lane_img, 0.5, 0.0)

        # FPS
        msg12      = str("FPS             " + str(int(1/time_diff)))
        combined_frame = cv2.putText(combined_frame,msg12,(50,270),
                                     cv2.FONT_HERSHEY_SIMPLEX,
                                     1,(0,255,0),2, cv2.LINE_AA)
        # steering angle phi
        msg145      = str("Steering angle  " + str(int(phi*180/3.14159)))
        combined_frame = cv2.putText(combined_frame,msg145,(50,200),
                                     cv2.FONT_HERSHEY_SIMPLEX,
                                     1,(0,255,0),2, cv2.LINE_AA)
        
        # ef
        msg1      = str("Lane error      " + str(int(ef)))
        combined_frame = cv2.putText(combined_frame,msg1,(50,235),
                                     cv2.FONT_HERSHEY_SIMPLEX,
                                     1,(0,255,0),2, cv2.LINE_AA)
        # Lane departure warning
        if abs(ef) > 90:
            msg1      = str('Warning: Lane departure')
            combined_frame = cv2.putText(combined_frame,msg1,(50,165),
                                     cv2.FONT_HERSHEY_SIMPLEX,
                                     1,(0,0,255),2, cv2.LINE_AA)

	# Check I/O
	print('\n')
	print('theta_e = ',theta_e)
	print('phi     = ',phi) 

	# To connect with other system and show inputs on screen
	if connect == 1:
		
		msg = str(n)
		sock.sendto(msg, (UDP_IP2, UDP_PORT2))
		print("frame no sent    ", msg) 

		data, addr  = sock.recvfrom(4004)               # stop/school sign comfirmation
		print('received msg ',data)
		print('\n')

		sign_msg = ''
		msg56    = ''
		if data == 'stop\n':
			sign_msg = 'STOP sign detected'
			msg56 = '   STOP'

		if data == 'school\n':
			sign_msg = 'SCHOOL sign detected'
			msg56 = 'Reduce speed' 

		# traffic sign
		combined_frame = cv2.putText(combined_frame,sign_msg,(50,310),
                                     cv2.FONT_HERSHEY_SIMPLEX,
                                     1,(0,0,255),2, cv2.LINE_AA)
		
		
		combined_frame = cv2.putText(combined_frame,msg56,(560,470),
                                     cv2.FONT_HERSHEY_SIMPLEX,
                                     4,(0,0,255),8, cv2.LINE_AA)

	# Display common HMI parameters
        combined_frame = cv2.circle(combined_frame,(lad_point),20,(255,255,255),-1)
        combined_frame = cv2.circle(combined_frame,lane_center,20,(0,255,0),-1)
        combined_frame = cv2.circle(combined_frame,(int(himg/2),vimg-10),20,(0,0,255),-1)
        combined_frame = cv2.line(combined_frame, lad_point, lane_center, (0, 255, 0), thickness=3, lineType=8)
        	
	cv2.namedWindow('Image processing',cv2.WINDOW_NORMAL)        
        cv2.resizeWindow('Image processing', 640,360)
        cv2.imshow('Image processing',img2)
	cv2.moveWindow("Image processing",20,440)
        
        cv2.namedWindow('Perspective view',cv2.WINDOW_NORMAL)        
        cv2.resizeWindow('Perspective view', 640,360)
        cv2.imshow('Perspective view',img3) # frame
	cv2.moveWindow("Perspective view",730,30)

	cv2.namedWindow('Result',cv2.WINDOW_NORMAL)        
        cv2.resizeWindow('Result',640,360)
        cv2.imshow('Result',combined_frame)
	cv2.moveWindow("Result",20,30)

	n        +=1
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()     










