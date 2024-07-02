# import pywavefront
# from pywavefront import visualization

# obj = pywavefront.Wavefront('Shelby.obj')

# visualization.draw(obj)

from PIL import Image
import json
import numpy as np
import model_input as model
import matplotlib.pyplot as plt
import enlighten
import sys
import pyrender
import trimesh


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


def createVoxelTable(mesh, resolution):
    print("Creating empty table")
    index = 0
    voxelList = []
    for z in range(resolution):
        yVoxelList = []
        for y in range(resolution):
            xVoxelList = []
            for x in range(resolution):
                xVoxelList.append(-1)
                index += 1
            yVoxelList.append(xVoxelList)
        voxelList.append(yVoxelList)

    print("Table created!")
    print("Filling with mesh data")
    index = 0
    for voxel in mesh:
        print_progress_bar(index, len(mesh), "Filling table:")
        voxelList[voxel["x"]][voxel["y"]][voxel["z"]] = voxel["color"]
    return voxelList


def hasNeighbourX(voxels, origin, resolution):
    neighbour = 0
    # print("origin : ", origin)
    # print("currentColor : ", voxels[origin[0]+neighbour][origin[1]][origin[2]])
    # print("next color : ", voxels[origin[0]+neighbour+1][origin[1]][origin[2]])
    while origin[0] + neighbour + 1 < resolution and voxels[origin[0] + neighbour + 1][origin[1]][origin[2]] != -1:
        neighbour += 1
    return neighbour


def hasNeighbourY(voxels, origin, neighbourX, resolution):
    neighbourY = 0
    while origin[1] + neighbourY + 1 < resolution:
        checkEveryX = 0
        for x in range(neighbourX):
            if origin[0] + x < resolution and voxels[origin[0] + x][origin[1] + neighbourY + 1][origin[2]] != -1:
                checkEveryX += 1
            else:
                break
        if checkEveryX == neighbourX:
            neighbourY += 1
        else:
            break
    return neighbourY


def hasNeighbourZ(voxels, origin, neighbourX, resolution):
    neighbourZ = 0
    while origin[2] + neighbourZ + 1 < resolution:
        checkEveryX = 0
        for x in range(neighbourX):
            if origin[0] + x < resolution and voxels[origin[0] + x][origin[1]][origin[2] + neighbourZ + 1] != -1:
                checkEveryX += 1
            else:
                break
        if checkEveryX == neighbourX:
            neighbourZ += 1
        else:
            break
    return neighbourZ


def removeVoxelsX(voxels, origin, number):
    for n in range(number):
        if origin[0] + n < resolution:
            voxels[origin[0] + n][origin[1]][origin[2]] = -1
    return voxels


def removeVoxelsXZ(voxels, origin, nX, nZ):
    for z in range(nZ):
        for x in range(nX):
            if origin[0] + x < resolution and origin[2] + z < resolution:
                voxels[origin[0] + x][origin[1]][origin[2] + z] = -1
    return voxels


def removeVoxelsXY(voxels, origin, nX, nY):
    for y in range(nY):
        for x in range(nX):
            if origin[0] + x < resolution and origin[1] + y < resolution:
                voxels[origin[0] + x][origin[1] + y][origin[2]] = -1
    return voxels


class Display:
    def __init__(self):
        self.scene = pyrender.Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))

    def addElement(self, element, resolution):
        index = 0
        voxelList = []

        boxf_trimesh = trimesh.creation.box(extents=1 * np.array([element["to"][0] - element["from"][0],
                                                                  element["to"][1] - element["from"][1],
                                                                  element["to"][2] - element["from"][2]]))
        boxf_face_colors = np.random.uniform(size=boxf_trimesh.faces.shape)
        boxf_trimesh.visual.face_colors = boxf_face_colors
        boxf_mesh = pyrender.Mesh.from_trimesh(boxf_trimesh, smooth=False)
        boxf_node = pyrender.Node(mesh=boxf_mesh,
                                  translation=np.array([element["from"][0], element["from"][1], element["from"][2]]))
        self.scene.add_node(boxf_node)

    def render(self, index):
        cam = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
        cam_pose = np.array([
            [0.0, -np.sqrt(2) / 2, np.sqrt(2) / 2, 0.5],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0.4],
            [0.0, 0.0, 0.0, 1.0]
        ])
        cam_node = self.scene.add(cam, pose=cam_pose)
        direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
        spot_l = pyrender.SpotLight(color=np.ones(3), intensity=10.0,
                                    innerConeAngle=np.pi / 16, outerConeAngle=np.pi / 6)
        point_l = pyrender.PointLight(color=np.ones(3), intensity=10.0)

        r = pyrender.OffscreenRenderer(viewport_width=640 * 2, viewport_height=480 * 2)
        color, depth = r.render(self.scene)
        r.delete()
        plt.figure(figsize=(20, 20))
        plt.imshow(color)
        plt.show()
        plt.close('all')

    def startRender(self):
        pyrender.Viewer(self.scene, use_raymond_lighting=True)


"""
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    voxelsArray = np.array(voxelList, ndmin=3)
    print("Np array shape is ", voxelsArray.shape)
    ax.voxels(voxelsArray, edgecolor='k')

    plt.show()
"""


def hasNeighbourZfromXY(voxels, origin, nX, nY, resolution):
    neighboursZ = 0
    while origin[2] + neighboursZ < resolution:
        validation = 0
        for y in range(nY):
            for x in range(nX):
                if voxels[origin[0] + x][origin[1] + y][origin[2]] != -1:
                    validation += 1
        if validation == nX * nY:
            neighboursZ += 1
        else:
            break
    return neighboursZ


def removeVoxelsXYZ(voxels, origin, nX, nY, nZ):
    for z in range(nZ):
        for y in range(nY):
            for x in range(nX):
                if origin[0] + x < resolution and origin[1] + y < resolution and origin[2] + z < resolution:
                    voxels[origin[0] + x][origin[1] + y][origin[2] + z] = -1
    return voxels


def printTexture(origin, texture, pixels, image):
    y = 0
    for xList in texture:
        x = 0
        for colorNumber in xList:
            image[origin[0] + x, origin[1] + y] = (
            pixels[colorNumber]["red"], pixels[colorNumber]["green"], pixels[colorNumber]["blue"])
            x += 1
        y += 1
    return image


def printElementTexture(origin, textures, pixels, image, maxWidth, lastHeight):
    if textures["totalWidth"] >= maxWidth:
        maxWidth = textures["totalWidth"]
        origin = (0, origin[1] + lastHeight)
        lastHeight = 0
    elif textures["totalWidth"] + origin[0] >= maxWidth:
        # print("LEEEEERRRRROOOOOYYYYY JEEEENNNNNKIIIIIINNNNNSS")
        origin = (0, origin[1] + lastHeight)
        lastHeight = 0

    image = printTexture(origin, textures["down"], pixels, image)
    origin = (origin[0] + len(textures["down"][0]), origin[1])
    if lastHeight < len(textures["down"]):
        lastHeight = len(textures["down"])
    image = printTexture(origin, textures["up"], pixels, image)
    origin = (origin[0] + len(textures["up"][0]), origin[1])
    if lastHeight < len(textures["up"]):
        lastHeight = len(textures["up"])
    image = printTexture(origin, textures["east"], pixels, image)
    origin = (origin[0] + len(textures["east"][0]), origin[1])
    if lastHeight < len(textures["east"]):
        lastHeight = len(textures["east"])
    image = printTexture(origin, textures["west"], pixels, image)
    origin = (origin[0] + len(textures["west"][0]), origin[1])
    if lastHeight < len(textures["west"]):
        lastHeight = len(textures["west"])
    image = printTexture(origin, textures["south"], pixels, image)
    origin = (origin[0] + len(textures["south"][0]), origin[1])
    if lastHeight < len(textures["south"]):
        lastHeight = len(textures["south"])
    image = printTexture(origin, textures["north"], pixels, image)
    origin = (origin[0] + len(textures["north"][0]), origin[1])
    if lastHeight < len(textures["north"]):
        lastHeight = len(textures["north"])

    return image, origin, lastHeight, maxWidth


def createElementTextures(voxels, image, pixels, origin, nX, nY, nZ, originOnImage, lastHeightOnImage, maxWidthOnImage,
                          width):
    textures = createTexture(voxels, origin, nX, nY, nZ)
    elementUVorigin = [originOnImage[0], originOnImage[1]]
    if textures["totalWidth"] + originOnImage[0] >= maxWidthOnImage:
        elementUVorigin[1] += lastHeightOnImage
        elementUVorigin[0] = 0
    image, originOnImage, lastHeightOnImage, maxWidthOnImage = printElementTexture(originOnImage, textures, pixels,
                                                                                   image, maxWidthOnImage,
                                                                                   lastHeightOnImage)

    element = {"__comment": "Cube", "from": [origin[0] * 48 / resolution - 16, origin[1] * 48 / resolution - 16,
                                             origin[2] * 48 / resolution - 16],
               "to": [origin[0] * 48 / resolution - 16 + 48 * nX / resolution,
                      origin[1] * 48 / resolution - 16 + 48 * nY / resolution,
                      origin[2] * 48 / resolution - 16 + 48 * nZ / resolution], "faces": {}}

    element["faces"]["north"] = {"uv": [elementUVorigin[0] * 16 / width, elementUVorigin[1] * 16 / width,
                                        elementUVorigin[0] * 16 / width + len(textures["down"][0]) * 16 / width,
                                        elementUVorigin[1] * 16 / width + len(textures["down"]) * 16 / width, ],
                                 "texture": "#texture"}
    elementUVorigin[0] += len(textures["down"][0])
    element["faces"]["south"] = {"uv": [elementUVorigin[0] * 16 / width, elementUVorigin[1] * 16 / width,
                                        elementUVorigin[0] * 16 / width + len(textures["up"][0]) * 16 / width,
                                        elementUVorigin[1] * 16 / width + len(textures["up"]) * 16 / width, ],
                                 "texture": "#texture"}
    elementUVorigin[0] += len(textures["up"][0])
    element["faces"]["down"] = {"uv": [elementUVorigin[0] * 16 / width, elementUVorigin[1] * 16 / width,
                                       elementUVorigin[0] * 16 / width + len(textures["east"][0]) * 16 / width,
                                       elementUVorigin[1] * 16 / width + len(textures["east"]) * 16 / width, ],
                                "texture": "#texture"}
    elementUVorigin[0] += len(textures["east"][0])
    element["faces"]["up"] = {"uv": [elementUVorigin[0] * 16 / width, elementUVorigin[1] * 16 / width,
                                     elementUVorigin[0] * 16 / width + len(textures["west"][0]) * 16 / width,
                                     elementUVorigin[1] * 16 / width + len(textures["west"]) * 16 / width, ],
                              "texture": "#texture"}
    elementUVorigin[0] += len(textures["west"][0])
    element["faces"]["east"] = {"uv": [elementUVorigin[0] * 16 / width, elementUVorigin[1] * 16 / width,
                                       elementUVorigin[0] * 16 / width + len(textures["south"][0]) * 16 / width,
                                       elementUVorigin[1] * 16 / width + len(textures["south"]) * 16 / width, ],
                                "texture": "#texture", "rotation": 90}
    elementUVorigin[0] += len(textures["south"][0])
    element["faces"]["west"] = {"uv": [elementUVorigin[0] * 16 / width, elementUVorigin[1] * 16 / width,
                                       elementUVorigin[0] * 16 / width + len(textures["north"][0]) * 16 / width,
                                       elementUVorigin[1] * 16 / width + len(textures["north"]) * 16 / width, ],
                                "texture": "#texture", "rotation": 90}
    elementUVorigin[0] += len(textures["north"][0])

    return image, element, originOnImage, lastHeightOnImage, maxWidthOnImage


"""
element = {"__comment": "Cube", "from": [origin[0] * 48 / resolution - 16, origin[1] * 48 / resolution - 16,
                                             origin[2] * 48 / resolution - 16],
               "to": [origin[0] * 48 / resolution - 16 + 48*nX / resolution,
                      origin[1] * 48 / resolution - 16 + 48*nY / resolution,
                      origin[2] * 48 / resolution - 16 + 48*nZ / resolution]}
    faces = {}
    for face in range(6):
        currentFace = {"uv": [originOnImage[0] * 16 / width, originOnImage[1] * 16 / width,
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
"""


def createTexture(voxels, origin, nX, nY, nZ):
    texture = {}
    down = []
    for y in range(nY):
        downX = []
        for x in range(nX):
            downX.append(voxels[origin[0] + x][origin[1] + y][origin[2]])
        down.append(downX)

    texture["down"] = down
    if nZ == 1:
        texture["up"] = down
    else:
        up = []
        for y in range(nY):
            upX = []
            for x in range(nX):
                upX.append(voxels[origin[0] + x][origin[1] + y][origin[2] + nZ - 1])
            up.append(upX)
        texture["up"] = up

    east = []
    for z in range(nZ):
        eastX = []
        for x in range(nX):
            eastX.append(voxels[origin[0] + x][origin[1] + nY - 1][origin[2] + z])
        east.append(eastX)

    texture["east"] = east

    if nY == 1:
        texture["west"] = east
    else:
        west = []
        for z in range(nZ):
            westX = []
            for x in range(nX):
                westX.append(voxels[origin[0] + x][origin[1]][origin[2] + z])
            west.append(westX)
        texture["west"] = west

    south = []
    for z in range(nZ):
        southY = []
        for y in range(nY):
            southY.append(voxels[origin[0]][origin[1] + y][origin[2] + z])
        south.append(southY)

    texture["south"] = south

    if nX == 1:
        texture["north"] = south
    else:
        north = []
        for z in range(nZ):
            northY = []
            for y in range(nY):
                northY.append(voxels[origin[0] + nX - 1][origin[1] + y][origin[2] + z])
            north.append(northY)
        texture["north"] = north
    texture["totalWidth"] = len(texture["down"][0]) + len(texture["up"][0]) + len(texture["east"][0]) + len(
        texture["west"][0]) + len(texture["south"][0]) + len(texture["north"][0])
    # print("total texture width = ", texture["totalWidth"])
    return texture


def createOptimizedElements(voxels, resolution, image, pixels, wdith):
    elements = []
    optimizedElements = 0
    basicElements = 0
    index = 0
    texture = {}
    maxWidth = 4096
    lastHeight = 0
    originOnImage = (0, 0)
    # displayVoxels(voxels, resolution)
    display = Display()

    for z in range(resolution):
        for y in range(resolution):
            for x in range(resolution):
                index += 1
                if voxels[x][y][z] != -1:
                    element = {}
                    neighboursX = hasNeighbourX(voxels, (x, y, z), resolution)
                    # print("nX = ", neighboursX)
                    neighboursY = hasNeighbourY(voxels, (x, y, z), neighboursX + 1, resolution)
                    # print("nY = ", neighboursY)
                    neighboursZ = hasNeighbourZ(voxels, (x, y, z), neighboursX + 1, resolution)
                    # print("nZ = ", neighboursZ)
                    # if neighboursX == 0 and neighboursY > 0:
                    # print("Ladys and gentlemen, we got him...")
                    if neighboursY != 0:
                        neighboursZfromXY = hasNeighbourZfromXY(voxels, (x, y, z), neighboursX + 1, neighboursY + 1,
                                                                resolution)
                        if neighboursZfromXY > 0:
                            image, element, originOnImage, lastHeight, maxWidth = createElementTextures(voxels, image,
                                                                                                        pixels,
                                                                                                        (x, y, z),
                                                                                                        neighboursX + 1,
                                                                                                        neighboursY + 1,
                                                                                                        neighboursZ + 1,
                                                                                                        originOnImage,
                                                                                                        lastHeight,
                                                                                                        maxWidth, width)
                            optimizedElements += 1
                            voxels = removeVoxelsXYZ(voxels, (x, y, z), neighboursX + 1, neighboursY + 1,
                                                     neighboursZ + 1)
                        else:
                            image, element, originOnImage, lastHeight, maxWidth = createElementTextures(voxels, image,
                                                                                                        pixels,
                                                                                                        (x, y, z),
                                                                                                        neighboursX + 1,
                                                                                                        neighboursY + 1,
                                                                                                        0 + 1,
                                                                                                        originOnImage,
                                                                                                        lastHeight,
                                                                                                        maxWidth, width)
                            optimizedElements += 1
                            voxels = removeVoxelsXY(voxels, (x, y, z), neighboursX + 1, neighboursY + 1)
                        # We have X and Y axis neighbours, that makes a rectangle, have to create an element with that

                    elif neighboursZ != 0:
                        image, element, originOnImage, lastHeight, maxWidth = createElementTextures(voxels, image,
                                                                                                    pixels,
                                                                                                    (x, y, z),
                                                                                                    neighboursX + 1,
                                                                                                    0 + 1,
                                                                                                    neighboursZ + 1,
                                                                                                    originOnImage,
                                                                                                    lastHeight,
                                                                                                    maxWidth, width)
                        optimizedElements += 1
                        voxels = removeVoxelsXZ(voxels, (x, y, z), neighboursX + 1, neighboursZ + 1)
                        # We have X and Z axis neighbours, that makes a rectangle, have to create an element with that

                    elif neighboursX != 0:
                        image, element, originOnImage, lastHeight, maxWidth = createElementTextures(voxels, image,
                                                                                                    pixels,
                                                                                                    (x, y, z),
                                                                                                    neighboursX + 1,
                                                                                                    0 + 1,
                                                                                                    0 + 1,
                                                                                                    originOnImage,
                                                                                                    lastHeight,
                                                                                                    maxWidth, width)
                        optimizedElements += 1
                        voxels = removeVoxelsX(voxels, (x, y, z), neighboursX + 1)
                        # We have X axis neighbours, that makes a 1D rectangle, have to create an element with that
                    else:
                        image, element, originOnImage, lastHeight, maxWidth = createElementTextures(voxels, image,
                                                                                                    pixels,
                                                                                                    (x, y, z),
                                                                                                    0 + 1,
                                                                                                    0 + 1,
                                                                                                    0 + 1,
                                                                                                    originOnImage,
                                                                                                    lastHeight,
                                                                                                    maxWidth, width)
                        voxels = removeVoxelsX(voxels, (x, y, z), neighboursX + 1)
                        basicElements += 1
                    elements.append(element)
                    display.addElement(element, resolution)
        print_progress_bar(index, resolution * resolution * resolution, "Optimizing:")
    print("Done ! Optimized elements : ", optimizedElements, " basic elements remaining : ", basicElements, " total : ",
          optimizedElements + basicElements)
    return image, elements


"""

    voxelList = []
    threads = list()
    with concurrent.futures.ThreadPoolExecutor(max_workers=256) as executor:
        for z in range(resolution):
            thread = executor.submit(createVoxelTable(mesh, resolution, z))
            threads.append(thread)
        for index, thread in enumerate(threads):
            voxelList.append(thread.result())
    return voxelList
"""

print("Starting !\nLoading the model and image pallette...")

mesh = model.lookup
pixels = model.pallette
texture = {}
# Creating the texture file

width = 2
while width * width < len(pixels):
    width = width * width
width = 4096
print("Texture image size for the model will be (", width, "x", width, ")")

img = Image.new("RGB", (width, width), "#FF00FF")

imgPixels = img.load()

resolution = 256

x = 0
y = 0
for pixel in pixels:
    if x >= width:
        x = 0
        y += 1

    imgPixels[x, y] = (pixel["red"], pixel["green"], pixel["blue"])

    x += 1

finalModel = {}
finalModel["__comment"] = "Converted from voxels to Minecraft by Sotshi"

finalModel["textures"] = {"texture": "texture_opti"}
finalModel["display"] = {
    "head": {
        "translation": [75.826, 54.259, 16],
        "scale": [4, 4, 4]
    }
}
finalModel["elements"] = []

print("Starting to convert voxel mesh into minecraft vanilla json model")

print("Converting mesh to easy access table...")
voxelArray = createVoxelTable(mesh, resolution)
image, elements = createOptimizedElements(voxelArray, resolution, imgPixels, pixels, width)

finalModel["elements"] = elements

img.show()
img.save("texture_opti.png", "PNG")

"""
iteration = 0
for voxel in mesh:
    if iteration%1000 == 0:
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
"""
print("Conversion done!\nSaving now")

with open('model_opti.json', 'w') as fp:
    json.dump(finalModel, fp, indent=4)

print("Saved!")
"""
img.show()
img.save("texture.png", "PNG")

"""
print("Saved texture!")
