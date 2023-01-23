import cv2
import numpy as np
 
RESIZED_IMAGE_WIDTH = 30
RESIZED_IMAGE_HEIGHT = 40

# Capital A
Chara = 65
imgTraininga = cv2.imread("images/A1.jpg")
imgGraya = cv2.cvtColor(imgTraininga, cv2.COLOR_BGR2GRAY)
retvala, imgTresha = cv2.threshold(imgGraya, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopya = imgTresha.copy()
imgContoursa, ha = cv2.findContours(imgTreshCopya, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital B
Charb = 66
imgTrainingb = cv2.imread("images/B1.jpg")
imgGrayB = cv2.cvtColor(imgTrainingb, cv2.COLOR_BGR2GRAY)
retvalB, imgTreshb = cv2.threshold(imgGrayB, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyb = imgTreshb.copy()
imgContoursb, hb = cv2.findContours(imgTreshCopyb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital C
Charc = 67
imgTrainingc = cv2.imread("images/C1.jpg")
imgGrayc = cv2.cvtColor(imgTrainingc, cv2.COLOR_BGR2GRAY)
retvalc, imgTreshc = cv2.threshold(imgGrayc, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyc = imgTreshc.copy()
imgContoursc, hc = cv2.findContours(imgTreshCopyc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital D
Chard = 68
imgTrainingd = cv2.imread("images/D1.jpg")
imgGrayd = cv2.cvtColor(imgTrainingd, cv2.COLOR_BGR2GRAY)
retvald, imgTreshd = cv2.threshold(imgGrayd, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyd = imgTreshd.copy()
imgContoursd, hd = cv2.findContours(imgTreshCopyd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital E
Chare = 69
imgTraininge = cv2.imread("images/E1.jpg")
imgGraye = cv2.cvtColor(imgTraininge, cv2.COLOR_BGR2GRAY)
retvale, imgTreshe = cv2.threshold(imgGraye, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopye = imgTreshe.copy()
imgContourse, he = cv2.findContours(imgTreshCopye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital F
Charf = 70
imgTrainingf = cv2.imread("images/F1.jpg")
imgGrayf = cv2.cvtColor(imgTrainingf, cv2.COLOR_BGR2GRAY)
retvalf, imgTreshf = cv2.threshold(imgGrayf, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyf = imgTreshf.copy()
imgContoursf, hf = cv2.findContours(imgTreshCopyf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital G
Charg = 71
imgTrainingg = cv2.imread("images/G1.jpg")
imgGrayg = cv2.cvtColor(imgTrainingg, cv2.COLOR_BGR2GRAY)
retvalg, imgTreshg = cv2.threshold(imgGrayg, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyg = imgTreshg.copy()
imgContoursg, hg = cv2.findContours(imgTreshCopyg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital H
Charh = 72
imgTrainingh = cv2.imread("images/H1.jpg")
imgGrayh = cv2.cvtColor(imgTrainingh, cv2.COLOR_BGR2GRAY)
retvalh, imgTreshh = cv2.threshold(imgGrayh, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyh = imgTreshh.copy()
imgContoursh, hh = cv2.findContours(imgTreshCopyh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital I
Chari = 73
imgTrainingi = cv2.imread("images/I1.jpg")
imgGrayi = cv2.cvtColor(imgTrainingi, cv2.COLOR_BGR2GRAY)
retvali, imgTreshi = cv2.threshold(imgGrayi, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyi = imgTreshi.copy()
imgContoursi, hi = cv2.findContours(imgTreshCopyi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital J
Charj = 74
imgTrainingj = cv2.imread("images/J1.jpg")
imgGrayj = cv2.cvtColor(imgTrainingj, cv2.COLOR_BGR2GRAY)
retvalj, imgTreshj = cv2.threshold(imgGrayj, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyj = imgTreshj.copy()
imgContoursj, hj = cv2.findContours(imgTreshCopyj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital K
Chark = 75
imgTrainingk = cv2.imread("images/K1.jpg")
imgGrayk = cv2.cvtColor(imgTrainingk, cv2.COLOR_BGR2GRAY)
retvalk, imgTreshk = cv2.threshold(imgGrayk, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyk = imgTreshk.copy()
imgContoursk, hk = cv2.findContours(imgTreshCopyk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital L
Charl = 76
imgTrainingl = cv2.imread("images/L1.jpg")
imgGrayl = cv2.cvtColor(imgTrainingl, cv2.COLOR_BGR2GRAY)
retvall, imgTreshl = cv2.threshold(imgGrayl, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyl = imgTreshl.copy()
imgContoursl, hl = cv2.findContours(imgTreshCopyl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital M
Charm = 77
imgTrainingm = cv2.imread("images/M1.jpg")
imgGraym = cv2.cvtColor(imgTrainingm, cv2.COLOR_BGR2GRAY)
retvalm, imgTreshm = cv2.threshold(imgGraym, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopym = imgTreshm.copy()
imgContoursm, hm = cv2.findContours(imgTreshCopym, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital N
Charn = 78
imgTrainingn = cv2.imread("images/N1.jpg")
imgGrayn = cv2.cvtColor(imgTrainingn, cv2.COLOR_BGR2GRAY)
retvaln, imgTreshn = cv2.threshold(imgGrayn, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyn = imgTreshn.copy()
imgContoursn, hn = cv2.findContours(imgTreshCopyn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital O
Charo = 79
imgTrainingo = cv2.imread("images/O1.jpg")
imgGrayo = cv2.cvtColor(imgTrainingo, cv2.COLOR_BGR2GRAY)
retvalo, imgTresho = cv2.threshold(imgGrayo, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyo = imgTresho.copy()
imgContourso, ho = cv2.findContours(imgTreshCopyo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital P
Charp = 80
imgTrainingp = cv2.imread("images/P1.jpg")
imgGrayp = cv2.cvtColor(imgTrainingp, cv2.COLOR_BGR2GRAY)
retvalp, imgTreshp = cv2.threshold(imgGrayp, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyp = imgTreshp.copy()
imgContoursp, hp = cv2.findContours(imgTreshCopyp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital Q
Charq = 81
imgTrainingq = cv2.imread("images/Q1.jpg")
imgGrayq = cv2.cvtColor(imgTrainingq, cv2.COLOR_BGR2GRAY)
retvalq, imgTreshq = cv2.threshold(imgGrayq, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyq = imgTreshq.copy()
imgContoursq, hq = cv2.findContours(imgTreshCopyq, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital R
Charr = 82
imgTrainingr = cv2.imread("images/R1.jpg")
imgGrayr = cv2.cvtColor(imgTrainingr, cv2.COLOR_BGR2GRAY)
retvalr, imgTreshr = cv2.threshold(imgGrayr, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyr = imgTreshr.copy()
imgContoursr, hr = cv2.findContours(imgTreshCopyr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital S
Chars = 83
imgTrainings = cv2.imread("images/S1.jpg")
imgGrays = cv2.cvtColor(imgTrainings, cv2.COLOR_BGR2GRAY)
retvals, imgTreshs = cv2.threshold(imgGrays, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopys = imgTreshs.copy()
imgContourss, hs = cv2.findContours(imgTreshCopys, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital T
Chart = 84
imgTrainingt = cv2.imread("images/T1.jpg")
imgGrayt = cv2.cvtColor(imgTrainingt, cv2.COLOR_BGR2GRAY)
retvalt, imgTresht = cv2.threshold(imgGrayt, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyt = imgTresht.copy()
imgContourst, ht = cv2.findContours(imgTreshCopyt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital U
Charu = 85
imgTrainingu = cv2.imread("images/U1.jpg")
imgGrayu = cv2.cvtColor(imgTrainingu, cv2.COLOR_BGR2GRAY)
retvalu, imgTreshu = cv2.threshold(imgGrayu, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyu = imgTreshu.copy()
imgContoursu, hu = cv2.findContours(imgTreshCopyu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital V
Charv = 86
imgTrainingv = cv2.imread("images/V1.jpg")
imgGrayv = cv2.cvtColor(imgTrainingv, cv2.COLOR_BGR2GRAY)
retvalv, imgTreshv = cv2.threshold(imgGrayv, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyv = imgTreshv.copy()
imgContoursv, hv = cv2.findContours(imgTreshCopyv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital W
Charw = 87
imgTrainingw = cv2.imread("images/W1.jpg")
imgGrayw = cv2.cvtColor(imgTrainingw, cv2.COLOR_BGR2GRAY)
retvalw, imgTreshw = cv2.threshold(imgGrayw, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyw = imgTreshw.copy()
imgContoursw, hw = cv2.findContours(imgTreshCopyw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital X
Charx = 88
imgTrainingy = cv2.imread("images/X1.jpg")
imgGrayy = cv2.cvtColor(imgTrainingy, cv2.COLOR_BGR2GRAY)
retvaly, imgTreshy = cv2.threshold(imgGrayy, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyy = imgTreshy.copy()
imgContoursy, hy = cv2.findContours(imgTreshCopyy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital Y
Chary = 89
imgTrainingy = cv2.imread("images/Y1.jpg")
imgGrayy = cv2.cvtColor(imgTrainingy, cv2.COLOR_BGR2GRAY)
retvaly, imgTreshy = cv2.threshold(imgGrayy, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyy = imgTreshy.copy()
imgContoursy, hy = cv2.findContours(imgTreshCopyy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Capital Z
Charz = 90
imgTrainingz = cv2.imread("images/Z1.jpg")
imgGrayz = cv2.cvtColor(imgTrainingz, cv2.COLOR_BGR2GRAY)
retvalz, imgTreshz = cv2.threshold(imgGrayz, 150, 255, cv2.CHAIN_APPROX_NONE)
imgTreshCopyz = imgTreshz.copy()
imgContoursz, hz = cv2.findContours(imgTreshCopyz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

flattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
intClassifications = []

for a in imgContoursb:
    [intX, intY, intW, intH] = cv2.boundingRect(a)
    imgROIa = imgTreshCopya[intY:intY + intH, intX:intX + intW]
    imgResizedROIa = cv2.resize(imgROIa, (RESIZED_IMAGE_WIDTH , RESIZED_IMAGE_HEIGHT))

    intClassifications.append(Chara)
    flatteningImg = imgResizedROIa.reshape(1, (RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    flattenedImages = np.append(flattenedImages, flatteningImg, 0)

    fltClassifications =np.array(intClassifications, np.float)
    finalClassifications = fltClassifications.reshape(fltClassifications.size, 1)


for b in imgContoursb:
    [intX, intY, intW, intH] = cv2.boundingRect(b)
    imgROIB = imgTreshCopyb[intY:intY + intH, intX:intX + intW]
    imgResizedROIB = cv2.resize(imgROIB, (RESIZED_IMAGE_WIDTH , RESIZED_IMAGE_HEIGHT))

    intClassifications.append(Charb)
    flatteningImg = imgResizedROIB.reshape(1, (RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    flattenedImages = np.append(flattenedImages, flatteningImg, 0)

for c in imgContoursc:
    [intX, intY, intW, intH] = cv2.boundingRect(c)
    imgROIc = imgTreshCopyc[intY:intY + intH, intX:intX + intW]
    imgResizedROIB = cv2.resize(imgROIB, (RESIZED_IMAGE_WIDTH , RESIZED_IMAGE_HEIGHT))

    intClassifications.append(Charb)
    flatteningImg = imgResizedROIB.reshape(1, (RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    flattenedImages = np.append(flattenedImages, flatteningImg, 0)

fltClassifications =np.array(intClassifications, np.float)
finalClassifications = fltClassifications.reshape(fltClassifications.size, 1)

np.savetxt("classifications.txt", finalClassifications)
np. savetxt("flattenedImages.txt",  flattenedImages)

print("complete")