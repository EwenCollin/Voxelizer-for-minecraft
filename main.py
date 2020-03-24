#import pywavefront
#from pywavefront import visualization

#obj = pywavefront.Wavefront('Shelby.obj')

#visualization.draw(obj)

from PIL import Image
import json

import model_input_low as model

import sys


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='='):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    # Print New Line on Complete
    if iteration == total:
        sys.stdout.write('\n')


def voxelTexX(number):
    x = 0
    y = 0
    index = 0
    for pixel in pixels:
        if x >= width:
            x = 0
            y += 1
        if index == number:
            return x
        index += 1
        x += 1

def voxelTexY(number):
    x = 0
    y = 0
    index = 0
    for pixel in pixels:
        if x >= width:
            x = 0
            y += 1
        if index == number:
            return y
        index += 1
        x += 1


print("Starting !\nLoading the model and image pallette...")

mesh = model.lookup
pixels = model.pallette

#Creating the texture file

width = 2
while width*width < len(pixels):
    width = width*width

print("Texture image size for the model will be (", width, "x", width, ")")

img = Image.new("RGB", (width, width), "#999999")

imgPixels = img.load()

resolution = 128

x = 0
y = 0
for pixel in pixels:
    if x >= width:
        x = 0
        y += 1

    imgPixels[x,y] = (pixel["red"], pixel["green"], pixel["blue"])

    x += 1

finalModel = {}
finalModel["__comment"] = "Converted from voxels to Minecraft by Sotshi"

finalModel["textures"] = {"texture": "texture"}

finalModel["elements"] = []

print("Starting to convert voxel mesh into minecraft vanilla json model")

iteration = 0
for voxel in mesh:
    print_progress_bar(iteration, len(mesh), "Conversion:", "")
    element = {"__comment": "Cube", "from": [voxel["x"] * 48 / resolution - 16, voxel["y"] * 48 / resolution - 16,
                                             voxel["z"] * 48 / resolution - 16],
               "to": [voxel["x"] * 48 / resolution - 16 + 48 / resolution,
                      voxel["y"] * 48 / resolution - 16 + 48 / resolution,
                      voxel["z"] * 48 / resolution - 16 + 48 / resolution]}
    faces = {}
    for face in range(6):
        currentFace = {"uv": [voxelTexX(voxel["color"]) * 16 / width, voxelTexY(voxel["color"]) * 16 / width,
                              voxelTexX(voxel["color"]) * 16 / width + 16 / width,
                              voxelTexY(voxel["color"]) * 16 / width + 16 / width], "texture": "#texture"}
        if face == 0:
            faces["down"] = currentFace
        if face == 1:
            faces["up"] = currentFace
        if face == 2:
            faces["north"] = currentFace
        if face == 3:
            faces["south"] = currentFace
        if face == 4:
            faces["west"] = currentFace
        if face == 5:
            faces["east"] = currentFace

    element["faces"] = faces
    finalModel["elements"].append(element)
    iteration += 1

print("Conversion done!\nSaving now")

with open('model.json', 'w') as fp:
    json.dump(finalModel, fp,  indent=4)

print("Saved!")

img.show()
img.save("texture.png", "PNG")


print("Saved texture!")

