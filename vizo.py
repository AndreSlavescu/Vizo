import cv2
import depthai as dai
import time
import os

stepSize = 0.05

newConfig = False

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

xoutDepth = pipeline.createXLinkOut()
xoutSpatialData = pipeline.createXLinkOut()
xinSpatialCalcConfig = pipeline.createXLinkIn()

xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

lrcheck = False
subpixel = False

stereo.initialConfig.setConfidenceThreshold(255)
stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)

# Config
topLeft = dai.Point2f(0.4, 0.4)
bottomRight = dai.Point2f(0.6, 0.6)

x = 0
configX = None

def findall(p, s):
    '''Yields all the positions of
    the pattern p in the string s.'''
    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i+1)

for i in range(15):
    j = 2
    topLeft1 = dai.Point2f(i/15 + 0.025, j/5 + 0.05)
    bottomRight1 = dai.Point2f(i/15 + 0.065, j/5 + 0.1)
    config1 = dai.SpatialLocationCalculatorConfigData()
    config1.depthThresholds.lowerThreshold = 50
    config1.depthThresholds.upperThreshold = 5000
    config1.roi = dai.Rect(topLeft1, bottomRight1)
    spatialLocationCalculator.initialConfig.addROI(config1)
    if i == 4 and j == 2:
        topLeft = topLeft1
        bottomRight = bottomRight1
        config = config1
        configX = x
    x += 1

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

    color = (0, 0, 0)
    prevTime = time.time()
    
    while True:
        inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived

        depthFrame = inDepth.getFrame()
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        
        xy = []
        minDist = 500

        spatialData = spatialCalcQueue.get().getSpatialLocations()

        dstr = ""
        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            depthMin = depthData.depthMin
            depthMax = depthData.depthMax

            xy.append((depthMin, depthMax))

            if depthMin > minDist:
                dstr += "F"
            else:
                dstr += "S"

            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

        # Show the frame
        cv2.imshow("depth", depthFrameColor)
        delta = time.time() - prevTime
        if delta > 1:
            print(dstr)
            findstr = "FFF"
            result = [(i, dstr[i:i+len(findstr)]) for i in findall(findstr, dstr)]

            minDist = 15
            midpoint = len(dstr)//2
            substrlen = len(findstr)
            
            for match in result:
                (i, _) = match
                (x1, x2) = (i, i+substrlen-1)
                distToCenter = abs(x1 + (x2 - x1)//2 - midpoint)
                if distToCenter < minDist:
                    minDist = distToCenter
                    minSpan = (x1, x2)

            if minDist < 15:
                (x1, x2) = minSpan
                distToCenter = x1 + (x2 - x1)//2 - midpoint
                if distToCenter < 0:
                    print("Move Left")
                    os.system("mpg321 -q left.mp3")
                elif distToCenter > 0:
                    print("Move Right")
                    os.system("mpg321 -q right.mp3")
                else:
                    print("Move Forward")
                    os.system("mpg321 -q forward.mp3")
            else:
                print("Turn Around")
                os.system("mpg321 -q around.mp3")

            prevTime = time.time()
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('w'):
            if topLeft.y - stepSize >= 0:
                topLeft.y -= stepSize
                bottomRight.y -= stepSize
                newConfig = True
        elif key == ord('a'):
            if topLeft.x - stepSize >= 0:
                topLeft.x -= stepSize
                bottomRight.x -= stepSize
                newConfig = True
        elif key == ord('s'):
            if bottomRight.y + stepSize <= 1:
                topLeft.y += stepSize
                bottomRight.y += stepSize
                newConfig = True
        elif key == ord('d'):
            if bottomRight.x + stepSize <= 1:
                topLeft.x += stepSize
                bottomRight.x += stepSize
                newConfig = True

        if newConfig:
            config.roi = dai.Rect(topLeft, bottomRight)
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)
            spatialCalcConfigInQueue.send(cfg)
            newConfig = False
