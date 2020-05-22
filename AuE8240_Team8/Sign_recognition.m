%% Initialization
clc;                                        %Clear the screen
clear;                                      %CLear workspace
close all;                                  %Close all open figures and images
%fclose(instrfindall);                       %Close all UDP comm channels

mov = VideoReader('video/Clockwise.mp4');         %Load the movie

%% UDP communication
ipA = '192.168.1.114';                      %IP of a remote PC
portA = 5006;                               %Port of remote PC
ipB = '192.168.1.65';                       %IP of a local PC
portB = 4004;                               %Port of local PC

udpB = udp(ipA,portA,'LocalPort',portB);    %UDP channel creation
fopen(udpB);                                %OPen UDP channel for communication
flushinput(udpB);                           %Flush the data in the comm channel

%% Create sign detector objects
detector_stop_sign = vision.CascadeObjectDetector('sign_data/Stop_sign_detector.xml');
detector_school_sign = vision.CascadeObjectDetector('sign_data/School_sign_detector.xml');

%% Frame by frame detection
while hasFrame(mov)
    
    pic1 = readFrame(mov);                              %Read the video frame
    pic = pic1(1:800,1:400,:);                          %Extract the region of interest for sign recognition
    
    bbox_stop = step(detector_stop_sign,pic);           %Detect a STOP sign
    bbox_school = step(detector_school_sign,pic);       %Detect a SCHOOL sign
    
%     %Display bounding box in the frame
%     detectedImg = insertObjectAnnotation(pic1,'rectangle',bbox_stop,'stop sign');
%     detectedImg = insertObjectAnnotation(detectedImg,'rectangle',bbox_school,'school sign');
%     imshow(detectedImg);
    
    while udpB.BytesAvailable <= 0                      %Hold the loop until frame index data is recieved
    end
    
    flushinput(udpB);                                   %Flush the channel after index confermation
    
    %Send the sign data to the remote PC
    if size(bbox_stop) > 0
        fprintf(udpB,'stop');
    end
    if size(bbox_school) > 0
        fprintf(udpB,'school');
    else
        fprintf(udpB,'continue');
    end
    
end