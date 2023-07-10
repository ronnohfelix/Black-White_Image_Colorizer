import numpy as np
import cv2
#credits to Neural Line
prototxt_path = "models/colorization_deploy_v2.prototxt"
model_path = "models/colorization_release_v2.caffemodel"
kernel_path = "models/pts_in_hull.npy"
image_path = r'C:\Users\KEN\Desktop\Black&White Images Colourizer\elon.jpg'

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
pts = np.load(kernel_path)

pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts.astype("float32")]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# LAB -> = Lightness a* b*
image = cv2.imread(image_path)
normalized_image = image.astype("float32") / 255.0
lab = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2LAB)

resized_image = cv2.resize(lab, (224, 224))
L = cv2.split(resized_image)[0]
L -= 50

net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
L = cv2.split(lab)[0]

colorized_image = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
colorized_image = cv2.cvtColor(colorized_image, cv2.COLOR_LAB2BGR)
colorized_image = (255 * colorized_image).astype("uint8")

cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()