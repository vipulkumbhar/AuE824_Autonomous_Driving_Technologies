negativeFolder = 'data\train4\Stop_negative';
trainCascadeObjectDetector('Stop_sign_detector.xml',gTruth,negativeFolder,'FalseAlarmRate',0.3,'NumCascadeStages',13);