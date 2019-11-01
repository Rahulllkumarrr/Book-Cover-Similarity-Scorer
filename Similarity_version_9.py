import numpy as np
import cv2
import os
from math import sqrt

cur=os.getcwd()
check=True
while check:

    loc1 = str(input("Enter Stock Image Location : \n"))
    stock = cv2.imread(cur + "/" + loc1, 0)




    while type(stock)==type(None):
        print("You have entered invalid file name.\n")
        loc1 = str(input("Enter Stock Image Location : \n"))
        stock = cv2.imread(cur + "/" + loc1, 0)
    stockSize = stock.shape[0]

    if stock.shape[0] > 900:
        ratio = 900 / stock.shape[0]
        stock = cv2.resize(stock, (0, 0), fx=ratio, fy=ratio)
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    # applying the sharpening kernel to the input image & displaying it.
    stock = cv2.filter2D(stock, -1, kernel_sharpening)

    loc2 = str(input("Enter Live Image Location :\n"))
    live = cv2.imread(cur + "/" + loc2, 0)



    while type(live) == type(None):
        print("You have entered invalid file name.\n")
        loc2 = str(input("Enter Live Image Location : \n"))
        live = cv2.imread(cur + "/" + loc2, 0)

    if live.shape[0] > 900:
        ratio = 900 / live.shape[0]
        live = cv2.resize(live, (0, 0), fx=ratio, fy=ratio)
    # if live.shape[0]>live.shape[1]:
    #     ratio = stock.shape[0] / live.shape[0]
    # else:
    #     ratio = stock.shape[0] / live.shape[1]



    def typeCol(val):
        if val<=127:
            return "dark"
        else:
            return "bright"


    # live = cv2.resize(live, (0, 0), fx=ratio, fy=ratio)

    rangeX=int(live.shape[1]/3.5)
    rangeY=int(live.shape[0]/3.5)

    U_L=(int(live.shape[1]/2)-rangeX,int(live.shape[0]/2)-rangeY)
    U_R=(int(live.shape[1]/2)+rangeX,int(live.shape[0]/2)-rangeY)
    L_L=(int(live.shape[1]/2)-rangeX,int(live.shape[0]/2)+rangeY)
    L_R=(int(live.shape[1]/2)+rangeX,int(live.shape[0]/2)+rangeY)

    outer=[U_L,U_R,L_L,L_R]

    liveBackgroundColour1=np.mean(live[0:25,0:25])
    liveBackgroundColour2=np.mean(live[-25:,0:25])
    liveBackgroundColour3=np.mean(live[-25:,-25:])
    liveBackgroundColour4=np.mean(live[0:25,-25:])

    meanBack=np.mean([liveBackgroundColour1,liveBackgroundColour2,liveBackgroundColour3,liveBackgroundColour4])

    bookColour=np.mean(live[int(live.shape[1]/2)-rangeX:int(live.shape[1]/2)+rangeX,int(live.shape[0]/2)-rangeY:int(live.shape[0]/2)+rangeY])


    # print("Book Colours are",typeCol(bookColour)," while Book Background colour is",typeCol(meanBack))



    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(live, -1, kernel_sharpening)

    BLURRING=52

    filtered=cv2.bilateralFilter(sharpened,25,BLURRING,52)

    blur=cv2.edgePreservingFilter(filtered, flags=1, sigma_s=60, sigma_r=0.3)


    # Apply edge detection method on the image
    edges = cv2.Canny(sharpened, 50, 150)


    corners = cv2.goodFeaturesToTrack(edges, 2000, 0.03, 10)
    corners = np.int0(corners)

    circle=live.copy()


    ULx=[]
    ULy=[]
    URx=[]
    URy=[]
    LLx=[]
    LLy=[]
    LRx=[]
    LRy=[]


    for corner in corners:
        x,y = corner.ravel()
        if x < U_L[0] and y < U_L[1]:
            ULx.append(x)
            ULy.append(y)
        elif x > U_R[0] and y < U_R[1]:
            URx.append(x)
            URy.append(y)
        elif x < L_L[0] and y > L_L[1]:
            LLx.append(x)
            LLy.append(y)
        elif x > L_R[0] and y > L_R[1]:
            LRx.append(x)
            LRy.append(y)
        cv2.circle(circle,(x,y),3,0,-1)
    # cv2.imshow("edges",edges)
    # cv2.imshow("sharp",sharpened)
    # cv2.imshow("circle",circle)
    # cv2.waitKey(0)
    if len(ULx)==0 or len(URx)==0 or len(LLx)==0 or len(LRx)==0:

        # print("Can't detect the Book Edges.Live image can't be cropped")
        warped=live
        kernel_sharpening = np.array([[-1, -1, -1],
                                      [-1, 9, -1],
                                      [-1, -1, -1]])
        # applying the sharpening kernel to the input image & displaying it.
        sharpened = cv2.filter2D(warped, -1, kernel_sharpening)
    else:
        UL=(min(ULx),min(ULy))
        UR=(max(URx),min(URy))
        LL=(min(LLx),max(LLy))
        LR=(max(LRx),max(LRy))



        cv2.circle(circle,UL,10,0,-1)
        cv2.circle(circle,UR,10,0,-1)
        cv2.circle(circle,LL,10,0,-1)
        cv2.circle(circle,LR,10,0,-1)
        UL = [min(ULx), min(ULy)]
        UR = [max(URx), min(URy)]
        LL = [min(LLx), max(LLy)]
        LR = [max(LRx), max(LRy)]

        point1=np.float32([UL,UR,LL,LR])
        trsns=np.float32([[0,0],[live.shape[1],0],[0,live.shape[0]],[live.shape[1],live.shape[0]]])
        # trsns=np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])
        # print(trsns)
        transformer=cv2.getPerspectiveTransform(point1,trsns)

        warped = cv2.warpPerspective(live.copy(), transformer, (live.shape[1], live.shape[0]))

        # if warped.shape[0] < 400 or warped.shape[1] < 400:
        #     kernel_sharpening = np.array([[-2, -2, -2],
        #                                   [-2, 18, -2],
        #                                   [-2, -2, -2]])
        # else:
        kernel_sharpening = np.array([[-1, -1, -1],
                                          [-1, 9, -1],
                                          [-1, -1, -1]])
        # applying the sharpening kernel to the input image & displaying it.
        sharpened = cv2.filter2D(warped, -1, kernel_sharpening)


    # live=sharpened
    cv2.imshow("live",sharpened)
    cv2.waitKey(0)
    #
    # BLURRING THE BOTH IMAGES
    stockBlur = cv2.blur(stock, (3, 3))
    liveBlur = cv2.blur(live, (3, 3))
    cropBlur= cv2.blur(sharpened,(3,3))

    # DETECTOR AND DESCRIPTOR
    detector = cv2.KAZE_create()

    # DETECTING  KEY POINTS AND FEATURES FROM INAGES
    kpsStock, featuresStock = detector.detectAndCompute(stock, None)
    kpsLive, featuresLive = detector.detectAndCompute(live, None)
    kpsCrop, featuresCrop = detector.detectAndCompute(sharpened, None)

    # Matcher
    matcher = cv2.BFMatcher(cv2.NORM_L2SQR,crossCheck=True)

    # MATCHING THE KEYPOINTS
    matches1 = matcher.match(featuresStock, featuresLive)
    matches2 = matcher.match(featuresStock, featuresCrop)

    # print(len(matches1),len(matches2))

    matches = max([len(matches1),len(matches2)])
    if len(matches1) < len(matches2):
        liveBlur=cropBlur

    stockMean=int(np.mean(stockBlur))
    liveMean=int(np.mean(liveBlur))
    diff=abs(stockMean-liveMean)

    # merged=cv2.drawMatches(stock,kpsStock,live,kpsLive,good[:10])
    maxi = min([len(featuresLive), len(featuresStock)])
    similarity = round(matches / maxi * 100, 2)
    # print(similarity)
    # low = round(len(low) / maxi * 100, 2)
    # if low>=50:
    if similarity>=50 and diff<100:
        # print("{} and {} are {}% Similar with {} stock mean and {} live mean".format(loc1,loc2,high,stockMean,liveMean))
        detector = cv2.KAZE_create()

        # DETECTING  KEY POINTS AND FEATURES FROM IMAGES
        kpsStock, featuresStock = detector.detectAndCompute(stockBlur, None)
        kpsLive, featuresLive = detector.detectAndCompute(liveBlur, None)

        # Matcher
        # matcher = cv2.BFMatcher(cv2.NORM_L2SQR,crossCheck=True)
        matcher = cv2.BFMatcher()

        # MATCHING THE KEYPOINTS
        matches = matcher.match(featuresStock, featuresLive)
        # print(len(featuresLive),len(featuresStock),len(matches))

        high = []

        for m in matches:
            if m.distance < 0.4:
                high.append([m])
        high=len(high)

        maxi = min([len(featuresLive), len(featuresStock)])
        if high>maxi:
            maxi=max([len(featuresLive), len(featuresStock)])
        percent = round(high / maxi * 100, 2)
        if 50 < percent < 75:
            percent = round(percent*1.25,2)
        elif percent <= 50:
            percent=round(percent*1.5,2)

    elif similarity>=40 and diff<100:
        # print("{} and {} are {}% Similar with {} stock mean and {} live mean".format(loc1,loc2,high,stockMean,liveMean))
        detector = cv2.KAZE_create()

        # DETECTING  KEY POINTS AND FEATURES FROM IMAGES
        kpsStock, featuresStock = detector.detectAndCompute(stockBlur, None)
        kpsLive, featuresLive = detector.detectAndCompute(liveBlur, None)

        # Matcher
        # matcher = cv2.BFMatcher(cv2.NORM_L2SQR,crossCheck=True)
        matcher = cv2.BFMatcher()

        # MATCHING THE KEYPOINTS
        matches = matcher.match(featuresStock, featuresLive)
        # print(len(featuresLive),len(featuresStock),len(matches))

        high = []

        for m in matches:
            if m.distance < 0.4:
                high.append([m])
        high = len(high)

        maxi = min([len(featuresLive), len(featuresStock)])
        if high > maxi:
            maxi = max([len(featuresLive), len(featuresStock)])
        percent = round(high / maxi * 100, 2)
        if 50 < percent < 70:
            percent = round(percent * 1.25, 2)
        elif percent <= 50:
            percent = round(percent * 1.5, 2)
    elif similarity>=35 and diff<100:
        detector = cv2.KAZE_create()

        # DETECTING  KEY POINTS AND FEATURES FROM IMAGES
        kpsStock, featuresStock = detector.detectAndCompute(stockBlur, None)
        kpsLive, featuresLive = detector.detectAndCompute(liveBlur, None)

        # Matcher
        matcher = cv2.BFMatcher(cv2.NORM_L2SQR,crossCheck=True)

        # MATCHING THE KEYPOINTS
        matches = matcher.match(featuresStock, featuresLive)
        # matcher = cv2.BFMatcher()
        # print(len(featuresLive),len(featuresStock),len(matches))

        high = []

        for m in matches:
            if m.distance < 0.4:
                high.append([m])
        high = len(high)
        maxi = min([len(featuresLive), len(featuresStock)])
        if high>maxi:
            maxi=max([len(featuresLive), len(featuresStock)])
        percent=round(high / maxi * 100, 2)
        if percent > 75:
            percent=round(percent/1.3,2)

    else:
        detector = cv2.KAZE_create()

        # DETECTING  KEY POINTS AND FEATURES FROM IMAGES
        kpsStock, featuresStock = detector.detectAndCompute(stockBlur, None)
        kpsLive, featuresLive = detector.detectAndCompute(liveBlur, None)

        # Matcher
        # matcher = cv2.BFMatcher(cv2.NORM_L2SQR,crossCheck=True)
        matcher = cv2.BFMatcher()
        # MATCHING THE KEYPOINTS
        matches = matcher.match(featuresStock, featuresLive)
        # print(len(featuresLive),len(featuresStock),len(matches))

        high = []

        for m in matches:
            if m.distance < 0.4:
                high.append([m])
        high = len(high)
        maxi = min([len(featuresLive), len(featuresStock)])
        if high>maxi:
            maxi=max([len(featuresLive), len(featuresStock)])
        percent = round(sqrt(high / maxi * 100), 2)

    # if percent<33:
    #     percent=round(sqrt(percent)*2,2)
    print("{}% Similar".format(percent))

    check = bool(int(input("0 to Exit 1 to Repeat")))
