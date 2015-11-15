from skimage.feature import hog

class HOG:
    def __init__(self, orientations = 9, pixelsPerCell = (8, 8), 
            cellsPerBlock = (3, 3), normalise = False):
        self.orientations = orientations
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.normalise = normalise

    def describe(self, image):
        hist = hog(image, orientations = self.orientations, pixels_per_cell = self.pixelsPerCell, cells_per_block = self.cellsPerBlock, normalise = self.normalise)
        return hist
