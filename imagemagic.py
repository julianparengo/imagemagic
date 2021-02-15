#!/usr/bin/env python
# coding: utf-8
import uuid
from typing import List, NamedTuple, Tuple
import enum
import random as rnd
import string
import tempfile
import os
import io
import sys
import argparse
import logging

import skimage as ski
import skimage.io
import skimage.measure
import skimage.filters
import skimage.draw
import skimage.morphology
import numpy as np
from PIL import Image
import shapely as sp
import shapely.geometry
import networkx as nx
import webcolors
import PDFNetPython3 as pdfnet
from PyPDF2 import PdfFileWriter, PdfFileReader
import cv2


LOGGER = logging.getLogger(__name__)


def random_string(n):
    return ''.join(rnd.choice(string.ascii_letters + string.digits) for _ in range(n))


def crop_to_content(image: np.ndarray) -> np.ndarray:
    """Crop to tightest rectangle that envelops the non-transparent part of the image."""
    if len(image.shape) != 3 or image.shape[2] != 4:
        raise ValueError("'image' should have an alpha-channel.")
    image = Image.fromarray(image).convert('RGBa')
    box = image.getbbox()
    image = image.crop(box)
    image = image.convert('RGBA')
    return np.array(image)


def resize_canvas(image: np.ndarray, size) -> np.ndarray:
    """Resize canvas without transforming the image, just like in Photoshop."""
    image = Image.fromarray(image)
    left_top_corner = ((size[0] - image.size[0]) // 2, (size[1] - image.size[1]) // 2)
    background = Image.new(image.mode, size)
    background.paste(image, left_top_corner)

    return np.array(background)


def compose_contours(contours: List[np.ndarray], size: tuple):
    """Draw image of 'contours', defined as polygons, as an image respecting nested contours (holes)."""
    contour_keys = {str(uuid.uuid4()): c for c in contours}

    # A multi-DAG where parent-child relationships mean that child contour is enclosed in the parent.
    contours_hierarchies = nx.MultiDiGraph()

    for key_a, contour_a in contour_keys.items():
        contours_hierarchies.add_node(key_a, contour=contour_a)
        for key_b, contour_b in contour_keys.items():
            if contour_a is contour_b:
                continue
            contours_hierarchies.add_node(key_b, contour=contour_b)
            if ski.measure.points_in_poly(contour_b, contour_a).all():
                contours_hierarchies.add_edge(key_a, key_b)

    # Get root nodes of the DAGs.
    roots = []
    for node in contours_hierarchies.nodes:
        if contours_hierarchies.in_degree(node) == 0:
            roots.append(node)

    # Final composite image where contours are going to be drawn with respect of "holes".
    mask = Image.new('1', size)

    # Now that's the tricky part.
    # Traverse each DAG of contours by descending down to each level of nesting
    # and either fill with color each contour on the level or fill with transparency
    # which changes each level starting with "fill with color".
    #
    # Fill the root contour, assume the next level to be "holes"
    # fill them with transparency
    # assume next level are "islands"
    # fill them with color
    # assume next level are "holes"
    # etc...
    stack = [(r, True) for r in roots]
    while stack:
        node, should_fill = stack.pop()
        contour = contours_hierarchies.nodes[node]['contour']
        contour_mask = ski.draw.polygon2mask((size[1], size[0]), contour)
        paste_mask = Image.fromarray(contour_mask)
        if not should_fill:
            contour_mask = np.logical_not(contour_mask)
        mask.paste(Image.fromarray(contour_mask), mask=paste_mask)
        stack.extend((n, not should_fill) for n in contours_hierarchies[node])

    return np.array(mask)


def color_to_rgb(color: str):
    """Convert color given as a name, hex or RGB to RGB."""
    try:
        return np.array(webcolors.name_to_rgb(color))
    except ValueError:
        pass
    try:
        return np.array(webcolors.hex_to_rgb(color))
    except ValueError:
        pass
    try:
        return np.array(webcolors.normalize_hex(color))
    except ValueError:
        raise ValueError(f'Impossible to interpret color {color}.')


def find_contours(image, offset: int = 0, smoothness: int = 0):
    """Find countours of content in an image with an alpha-channel."""
    # So that the outline won't go beyond the canvas.
    image = resize_canvas(image, [offset * 3 + e for e in (image.shape[1], image.shape[0])])
    ret, mask = cv2.threshold(image[:, :, 3], 0, 255, cv2.THRESH_BINARY)
    cvcontours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cvcontours, (image.shape[1], image.shape[0])
  
    """
    alpha = np.dsplit(image, 4)[3].reshape(image.shape[:2])
    alpha = alpha / alpha.max()  # Normalize to range [0, 1].
    
    if offset != 0:
        selem = ski.morphology.disk(offset)
        if offset > 0:
            LOGGER.debug('start morph dilation Line 136...')
            #alpha = ski.morphology.dilation(alpha, selem)
            ret, mask = cv2.threshold(image[:, :, 3], 0, 255, cv2.THRESH_BINARY)
            cvcontours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return cvcontours, (image.shape[1], image.shape[0])
        else:
            LOGGER.debug('start morph erosion Line 139...')
            alpha = ski.morphology.erosion(alpha, selem)

    if smoothness != 0:
        alpha = ski.filters.gaussian(alpha, sigma=smoothness)

    contours = ski.measure.find_contours(alpha, 0.5)
    return contours, (image.shape[1], image.shape[0])
  """

def colorize(image, color):
    """Convert 1-bit image to RGBA."""
    transparent_pixel = np.array([0, 0, 0, 0])
    color_pixel = np.append(color_to_rgb(color), 255)
    result = np.empty(image.shape[:2] + (4,), dtype='uint8')
    result[image == 0] = transparent_pixel
    result[image != 0] = color_pixel
    return result


def stack(*images: np.ndarray):
    """Stack images on top of each other."""
    if not images:
        raise ValueError("No images provided.")

    result = Image.fromarray(images[0])
    for image in images[1:]:
        image = Image.fromarray(image)
        if image.size != result.size:
            raise ValueError('Images should have the same shape.')
        result.paste(image, (0, 0), image)
    return np.array(result)


class FitMethod(enum.Enum):
    CROP = 'CROP'
    EXPAND = 'EXPAND'


def fit(image: np.ndarray, size: Tuple[int, int], method: FitMethod = FitMethod.CROP):
    """Fit 'image' into 'shape'."""
    image = Image.fromarray(image)
    xy_ratio_image = image.size[0] / image.size[1]
    xy_ratio_shape = size[0] / size[1]

    by_smallest = True
    if method == FitMethod.EXPAND:
        by_smallest = False

    if by_smallest:
        condition = xy_ratio_image >= xy_ratio_shape
    else:
        condition = xy_ratio_image < xy_ratio_shape

    if condition:
        new_shape = (
            int(np.ceil(image.size[0] * size[1] / image.size[1])), size[1]
        )
    else:
        new_shape = (
            size[0], int(np.ceil(image.size[1] * size[0] / image.size[0]))
        )
    image = image.resize(new_shape)
    if method == FitMethod.CROP:
        box = [0, 0, image.size[0], image.size[1]]
        if image.size[0] > size[0]:
            box[0] += int(np.floor((image.size[0] - size[0]) / 2))
            box[2] -= int(np.ceil((image.size[0] - size[0]) / 2))
        if image.size[1] > size[1]:
            box[1] += int(np.floor((image.size[1] - size[1]) / 2))
            box[3] -= int(np.ceil((image.size[1] - size[1]) / 2))
        image = image.crop(box)
    elif method == FitMethod.EXPAND:
        image = Image.fromarray(resize_canvas(np.array(image), size))

    return np.array(image)


def contours_to_svg(contours: np.ndarray, size) -> str:
    """Convert contours as returned by find_contours() into an SVG image."""
    result = f'<svg width="{size[0]}" height="{size[1]}" xmlns="http://www.w3.org/2000/svg">'

    for contour in contours:
        result += '<path d="M'
        for x, y in contour:
            result += f"{y} {x} "
        if contour.size > 0:
            result += f"{contour[0][1]} {contour[0][0]} "
        result += '" style="fill:none; stroke:#000000; stroke-width:1"/>'
    result += "</svg>"
    return result


def resize_contours(contours, original_shape, new_shape):
    """Resize 'contours' to fit into new canvas of different 'shape'."""
    x_ratio = new_shape[0] / original_shape[0]
    y_ratio = new_shape[1] / original_shape[1]
    new_contours = []
    for contour in contours:
        contour = contour.copy()
        contour[:, 0] = (contour[:, 0] * y_ratio)
        contour[:, 1] = (contour[:, 1] * x_ratio)
        new_contours.append(contour)
    return new_contours


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIBUS_ROOT = os.environ["SCRIBUS_ROOT"] = "C:\Program Files\Scribus 1.5.1"


# A utility function used to add new Content Groups (Layers) to the document.
def CreateLayer(doc, layer_name):
    grp = pdfnet.Group.Create(doc, layer_name)
    cfg = doc.GetOCGConfig()
    if not cfg.IsValid():
        cfg = pdfnet.Config.Create(doc, True)
        cfg.SetName("Default")

    # Add the new OCG to the list of layers that should appear in PDF viewer GUI.
    layer_order_array = cfg.GetOrder()
    if layer_order_array is None:
        layer_order_array = doc.CreateIndirectArray()
        cfg.SetOrder(layer_order_array)
    layer_order_array.PushBack(grp.GetSDFObj())
    return grp


def CreateGroup(doc, layer, path, width, height):
    writer = pdfnet.ElementWriter()
    writer.Begin(doc.GetSDFDoc())

    # Create an Image that can be reused in the document or on the same page.
    img = pdfnet.Image.Create(doc.GetSDFDoc(), path)
    builder = pdfnet.ElementBuilder()
    element = builder.CreateImage(img, pdfnet.Matrix2D(width, 0, 0, height, 0, 0))
    writer.WritePlacedElement(element)

    grp_obj = writer.End()

    # Indicate that this form (content group) belongs to the given layer (OCG).
    grp_obj.PutName("Subtype", "Form")
    grp_obj.Put("OC", layer)
    grp_obj.PutRect("BBox", 0, 0, img.GetImageWidth(), img.GetImageHeight())  # Set the clip box for the content.

    return grp_obj


# Creates some content (a path in the shape of a heart) and associate it with the vector layer
def CreatePath(doc, layer, contour, width, height):
    writer = pdfnet.ElementWriter()
    writer.Begin(doc.GetSDFDoc())

    # Create a path object in the shape of a heart
    builder = pdfnet.ElementBuilder()
    builder.PathBegin()  # start constructing the path

    begin_pt = contour[0]
    builder.MoveTo(float(begin_pt[0][0]), float(height - begin_pt[0][1]))
    cur_pt = begin_pt
    for i in range(1, len(contour)):
        if i % 2 != 0:
            continue
        cur_pt = contour[i]
        builder.LineTo(float(cur_pt[0][0]), float(height - cur_pt[0][1]))

    builder.LineTo(float(begin_pt[0][0]), float(height - begin_pt[0][1]))
    builder.ClosePath()
    element = builder.PathEnd()  # the path geometry is now specified.

    # Set the path STROKE color space and color
    element.SetPathStroke(True)
    gstate = element.GetGState()
    gstate.SetStrokeColorSpace(pdfnet.ColorSpace.CreateDeviceRGB())
    gstate.SetStrokeColor(pdfnet.ColorPt(0.0, 0.0, 0.0))
    # gstate.SetStrokeColorSpace(ColorSpace.CreateDeviceCMYK())
    # gstate.SetStrokeColor(ColorPt(0.0, 1.0, 0.0, 0.0))
    gstate.SetLineWidth(1)

    writer.WriteElement(element)

    grp_obj = writer.End()

    # Indicate that this form (content group) belongs to the given layer (OCG).
    grp_obj.PutName("Subtype", "Form")
    grp_obj.Put("OC", layer)
    grp_obj.PutRect("BBox", 0, 0, width, height)  # Set the clip box for the content.

    return grp_obj


def MakePDF(input_path, width, height, out_path, contour):
    pdfnet.PDFNet.Initialize()

    # Create three layers...
    doc = pdfnet.PDFDoc()
    image_layer = pdfnet.CreateLayer(doc, "Image Layer")
    contour_layer = pdfnet.CreateLayer(doc, "Contour Layer")

    # Start a new page ------------------------------------
    page = doc.PageCreate(pdfnet.Rect(0, 0, width, height))

    builder = pdfnet.ElementBuilder()  # ElementBuilder is used to build new Element objects
    writer = pdfnet.ElementWriter()  # ElementWriter is used to write Elements to the page
    writer.Begin(page)  # Begin writting to the page

    # Add new content to the page and associate it with one of the layers.
    element = builder.CreateForm(CreateGroup(doc, image_layer.GetSDFObj(), input_path, width, height))
    writer.WriteElement(element)

    element = builder.CreateForm(CreatePath(doc, contour_layer.GetSDFObj(), contour, width, height))
    writer.WriteElement(element)

    writer.End()  # save changes to the current page
    doc.PagePushBack(page)
    # Set the default viewing preference to display 'Layer' tab
    prefs = doc.GetViewPrefs()
    prefs.SetPageMode(pdfnet.PDFDocViewPrefs.e_UseOC)

    doc.Save(out_path, pdfnet.SDFDoc.e_linearized)
    doc.Close()
    print(out_path)


def MakeSVGFromContour(contour, width, height):
    with open("path.svg", "w+") as f:
        f.write('<?xml version="1.0"?>')
        f.write('<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"')
        f.write('  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">')

        f.write(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">')
        f.write('<path d="M')
        x0, y0 = contour[0][0]
        for i in range(len(contour)):
            if i % 2 != 0:
                continue
            x, y = contour[i][0]
            f.write(f"{x} {y} ")

        f.write(f"{x0} {y0} ")

        f.write('" style="fill:none; stroke:#000000; stroke-width:1"/>')
        f.write("</svg>")

    return


def ConvertSVG2PDF():
    py_path = APP_ROOT + "\\test.py"
    in_path = APP_ROOT + "\\path.svg"
    out_path = APP_ROOT + "\\path.pdf"
    exe_path = f'"{SCRIBUS_ROOT}\Scribus"' + " -g -ns -py {} {} {}".format(py_path, in_path, out_path)
    os.system(exe_path)

    return


def MergePdf(input_path, width, height, out_path):
    pdfnet.PDFNet.Initialize()

    # Create three layers...
    doc = pdfnet.PDFDoc(APP_ROOT + "\\path.pdf")
    image_layer = CreateLayer(doc, "Image Layer")

    page = doc.GetPage(1)

    builder = pdfnet.ElementBuilder()  # ElementBuilder is used to build new Element objects
    writer = pdfnet.ElementWriter()  # ElementWriter is used to write Elements to the page
    writer.Begin(page)  # Begin writting to the page

    # Add new content to the page and associate it with one of the layers.
    element = builder.CreateForm(CreateGroup(doc, image_layer.GetSDFObj(), input_path, width, height))
    writer.WriteElement(element)

    writer.End()  # save changes to the current page
    doc.PagePushBack(page)
    # Set the default viewing preference to display 'Layer' tab
    prefs = doc.GetViewPrefs()
    prefs.SetPageMode(pdfnet.PDFDocViewPrefs.e_UseOC)

    doc.Save(out_path, pdfnet.SDFDoc.e_linearized)
    doc.Close()
    print(out_path)


class OutputType(enum.Enum):
    PDF = 'PDF'
    PNG = 'PNG'


class ResizeMethod:
    BY_LONGEST_SIDE = 'BY_LONGEST_SIDE'
    EXACT = 'EXACT'


class OutputSetting(NamedTuple):
    type: OutputType
    size: Tuple[int, int]
    name_pattern: str
    watermark: bool = False
    background: bool = False
    outline: bool = False
    outline_color: str = 'white'
    outline_smoothness: int = 15
    outline_offset: int = 40
    resize_method: ResizeMethod = ResizeMethod.BY_LONGEST_SIDE


OUTPUT_SETTINGS = [
    OutputSetting(
        type=OutputType.PNG,
        size=(475, 475),
        name_pattern='{prefix}{index:03d}.png',
        watermark=True,
        background=True,
        outline=True,
        resize_method=ResizeMethod.EXACT
    ),
    OutputSetting(
        type=OutputType.PNG,
        size=(1216, 1216),
        name_pattern='{prefix}{index:03d}SM.png',
        watermark=False,
        background=False,
        outline_offset=0,
    ),
    OutputSetting(
        type=OutputType.PNG,
        size=(1649, 1649),
        name_pattern='{prefix}{index:03d}MD.png',
        watermark=False,
        background=False,
        outline_offset=0,
    ),
    OutputSetting(
        type=OutputType.PNG,
        size=(2081, 2081),
        name_pattern='{prefix}{index:03d}LG.png',
        watermark=False,
        background=False,
        outline_offset=0,
    ),
    OutputSetting(
        type=OutputType.PDF,
        size=(1216, 1216),
        name_pattern='{prefix}{index:03d}SM.pdf',
        watermark=False,
        background=False,
    ),
    OutputSetting(
        type=OutputType.PDF,
        size=(1649, 1649),
        name_pattern='{prefix}{index:03d}MD.pdf',
        watermark=False,
        background=False,
    ),
    OutputSetting(
        type=OutputType.PDF,
        size=(2081, 2081),
        name_pattern='{prefix}{index:03d}LG.pdf',
        watermark=False,
        background=False,
    ),
]


def main():
    parser = argparse.ArgumentParser('Process images.')
    parser.add_argument('input_dir')
    parser.add_argument('png_output_dir')
    parser.add_argument('pdf_output_dir')
    parser.add_argument('background')
    parser.add_argument('watermark')
    parser.add_argument('prefix', default='')
    arguments = parser.parse_args()
    for path in [arguments.png_output_dir, arguments.pdf_output_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

    for idx, entry in enumerate(os.scandir(arguments.input_dir)):
        LOGGER.debug('Processing %s...', entry.name)
        if not entry.name.endswith(".png"):
            continue
        input_image = ski.io.imread(entry.path)
        if input_image.shape[2] < 4:
            LOGGER.warning("PNG format error : there is no alpha channel in file %s.", entry.name)
            continue
        for settings in OUTPUT_SETTINGS:
            image = input_image.copy()
            image = crop_to_content(image)

            if settings.resize_method == ResizeMethod.EXACT:
                settings_ratio = settings.size[0] / settings.size[1]
                image_ratio = image.shape[1] / image.shape[0]
                if image_ratio > settings_ratio:
                    new_size = (image.shape[1], int(np.ceil(image.shape[1] * settings_ratio)))
                elif image_ratio < settings_ratio:
                    new_size = (int(np.ceil(image.shape[0] / settings_ratio)), image.shape[0])
                else:
                    new_size = image.shape[1], image.shape[0]
                LOGGER.debug(
                    'Adjusting main image\'s aspect ratio from %s to %s. Old size %s, new size %s...',
                    image_ratio,
                    settings_ratio,
                    f'{image.shape[1]}x{image.shape[0]}',
                    f'{new_size[0]}x{new_size[1]}',
                )
                image = resize_canvas(image, new_size)
            LOGGER.debug('Finding content contours...')
            contours, size = find_contours(image, settings.outline_offset, settings.outline_smoothness
            )
            if settings.outline:
                LOGGER.debug('Making outline...')
                image = resize_canvas(image, size)
                contour = compose_contours(contours, size)
                outline = colorize(contour, settings.outline_color)
                image = stack(outline, image)
            if settings.background:
                LOGGER.debug('Making background...')
                background = ski.io.imread(arguments.background)
                background = fit(background, (image.shape[1], image.shape[0]))
                image = stack(background, image)
            if settings.watermark:
                LOGGER.debug('Making watermark...')
                watermark = ski.io.imread(arguments.watermark)
                watermark = fit(watermark, (image.shape[1], image.shape[0]))
                image = stack(image, watermark)

            output_name = settings.name_pattern.format(
                prefix=arguments.prefix,
                index=idx
            )
            LOGGER.debug('Resizing...')
            if settings.type == OutputType.PDF:
                if not settings.outline:
                    image = resize_canvas(image, size)
            if settings.resize_method == ResizeMethod.EXACT:
                new_size = settings.size
                if settings.type == OutputType.PDF:
                    contours = resize_contours(contours, size, settings.size)
            elif settings.resize_method == ResizeMethod.BY_LONGEST_SIDE:
                ratio = max(settings.size) / max(image.shape[:2])
                new_size = (int(np.ceil(image.shape[1] * ratio)), int(np.ceil(image.shape[0] * ratio)))
                if settings.type == OutputType.PDF:
                    contours = resize_contours(contours[0], size, new_size)
            else:
                raise RuntimeError
            image = np.array(Image.fromarray(image).resize(new_size))

            if settings.type == OutputType.PNG:
                LOGGER.debug('Writing PNG...')
                output_path = os.path.join(arguments.png_output_dir, output_name)
                Image.fromarray(image).save(output_path)
            elif settings.type == OutputType.PDF:
                LOGGER.debug('Writing PDF...')
                output_path = os.path.join(arguments.pdf_output_dir, output_name)
                svg = contours_to_svg(contours, new_size)
                open(f'{APP_ROOT}\\path.svg', 'w').write(svg)
                tmp_image = 'tmp.png'
                skimage.io.imsave(tmp_image, image)
                ConvertSVG2PDF()
                MergePdf(tmp_image, new_size[0], new_size[1], output_path)

                # MakePDF(input_path, width, height, out_path, contour)

                infile = PdfFileReader(output_path, 'rb')
                output = PdfFileWriter()
                p = infile.getPage(1)
                output.addPage(p)
                with open(output_path, 'wb') as f:
                    output.write(f)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main()
