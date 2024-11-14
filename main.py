import matplotlib.pyplot as plt
import numpy as np

image_path = "image.jpg"
img = plt.imread(image_path)

#grayscale by averaging the RGB values
gray_img = np.mean(img[..., :3], axis=2)

#resize the image
resized_img = img[::4, ::4]

#apply a basic blur by averaging a 3x3 neighborhood(also grays out the image)
def apply_blur(image):
    #create an empty array to store the blurred image
    blurred = np.zeros_like(image)
    #apply a simple 3x3 averaging kernel
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            blurred[i, j] = np.mean(image[i-1:i+2, j-1:j+2])
    return blurred

blurred_img = apply_blur(img)

#edge detection
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

def apply_sobel(image, kernel):
    #edge pixels
    edge_image = np.zeros_like(image)
    #iterating over each pixel skipping borders
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            region = image[i-1:i+2, j-1:j+2]
            edge_image[i, j] = np.sum(region * kernel)#gives dot product which is = intensity of edge pixel
    return edge_image

edges_x = apply_sobel(gray_img, sobel_x)#horizontal gradients
edges_y = apply_sobel(gray_img, sobel_y)#vertical gradients
edges = np.sqrt(edges_x**2 + edges_y**2)#euclidean formula to get edge intensity by combining gradients







#main display
fig, axes = plt.subplots(1, 5, figsize=(16, 4))
axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[1].imshow(gray_img, cmap="gray")
axes[1].set_title("Grayscale Image")
axes[2].imshow(resized_img, cmap="gray")
axes[2].set_title("Resized Image")
axes[3].imshow(edges, cmap="gray")
axes[3].set_title("Edge Detection")
axes[4].imshow(blurred_img)
axes[4].set_title("Blurred Image")
for ax in axes:
    ax.axis("off")
plt.show()
