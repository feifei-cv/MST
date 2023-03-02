import numpy as np
import cv2


def color_transfer(source, target):

    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar
    epsilon = 1e-30
    # scale by the standard deviations
    l = ( lStdSrc / (lStdTar+epsilon)) * l
    a = ( aStdSrc / (aStdTar+epsilon)) * a
    b = ( bStdSrc / (bStdTar+epsilon)) * b
    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc
    # clip the pixel intensities to [0, 255] if they fall outside
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    return transfer


def image_stats(image):

    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())
    return (lMean, lStd, aMean, aStd, bMean, bStd)


