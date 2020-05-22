negativeFolder = 'data\train4\School_negative';
trainCascadeObjectDetector('School_sign_detector.xml',gTruth,negativeFolder,'FalseAlarmRate',0.3,'NumCascadeStages',13);