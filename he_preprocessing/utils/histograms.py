import colorsys
import numpy as np
from PIL import ImageFont, ImageDraw, Image

import image as image_util
from matplotlib import pyplot as plt

from he_preprocessing.constants import SCALE_FACTOR, TILE_TEXT_W_BORDER, TILE_TEXT_H_BORDER, FONT_PATH, TILE_TEXT_SIZE, \
    TILE_TEXT_COLOR, TILE_TEXT_BACKGROUND_COLOR


def np_hsv_hue_histogram(h):
    """
    Create Matplotlib histogram of hue values for an HSV image and return the histogram as a NumPy array image.

    Args:
      h: Hue values as a 1-dimensional int NumPy array (scaled 0 to 360)

    Returns:
      Matplotlib histogram of hue values converted to a NumPy array image.
    """
    figure = plt.figure()
    canvas = figure.canvas
    _, _, patches = plt.hist(h, bins=360)
    plt.title("HSV Hue Histogram, mean=%3.1f, std=%3.1f" % (np.mean(h), np.std(h)))

    bin_num = 0
    for patch in patches:
        rgb_color = colorsys.hsv_to_rgb(bin_num / 360.0, 1, 1)
        # patch.set_facecolor(rgb_color)
        bin_num += 1

    canvas.draw()
    w, h = canvas.get_width_height()
    np_hist = np.fromstring(canvas.get_renderer().tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(figure)
    image_util.np_info(np_hist)
    return np_hist


def np_histogram(data, title, bins="auto"):
    """
   Create Matplotlib histogram and return it as a NumPy array image.

   Args:
     data: Data to plot in the histogram.
     title: Title of the histogram.
     bins: Number of histogram bins, "auto" by default.

   Returns:
     Matplotlib histogram as a NumPy array image.
   """
    figure = plt.figure()
    canvas = figure.canvas
    plt.hist(data, bins=bins)
    plt.title(title)

    canvas.draw()
    w, h = canvas.get_width_height()
    np_hist = np.fromstring(canvas.get_renderer().tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(figure)
    image_util.np_info(np_hist)
    return np_hist


def np_hsv_saturation_histogram(s):
    """
    Create Matplotlib histogram of saturation values for an HSV image and return the histogram as a NumPy array image.

    Args:
      s: Saturation values as a 1-dimensional float NumPy array

    Returns:
      Matplotlib histogram of saturation values converted to a NumPy array image.
    """
    title = "HSV Saturation Histogram, mean=%.2f, std=%.2f" % (np.mean(s), np.std(s))
    return np_histogram(s, title)


def np_hsv_value_histogram(v):
    """
    Create Matplotlib histogram of value values for an HSV image and return the histogram as a NumPy array image.

    Args:
      v: Value values as a 1-dimensional float NumPy array

    Returns:
      Matplotlib histogram of saturation values converted to a NumPy array image.
    """
    title = "HSV Value Histogram, mean=%.2f, std=%.2f" % (np.mean(v), np.std(v))
    return np_histogram(v, title)


def np_rgb_channel_histogram(rgb, ch_num, ch_name):
    """
    Create Matplotlib histogram of an RGB channel for an RGB image and return the histogram as a NumPy array image.

    Args:
      rgb: Image as RGB NumPy array.
      ch_num: Which channel (0=red, 1=green, 2=blue)
      ch_name: Channel name ("R", "G", "B")

    Returns:
      Matplotlib histogram of RGB channel converted to a NumPy array image.
    """

    ch = rgb[:, :, ch_num]
    ch = ch.flatten()
    title = "RGB %s Histogram, mean=%.2f, std=%.2f" % (ch_name, np.mean(ch), np.std(ch))
    return np_histogram(ch, title, bins=256)


def np_rgb_r_histogram(rgb):
    """
    Obtain RGB R channel histogram as a NumPy array image.

    Args:
      rgb: Image as RGB NumPy array.

    Returns:
       Histogram of RGB R channel as a NumPy array image.
    """
    hist = np_rgb_channel_histogram(rgb, 0, "R")
    return hist


def np_rgb_g_histogram(rgb):
    """
    Obtain RGB G channel histogram as a NumPy array image.

    Args:
      rgb: Image as RGB NumPy array.

    Returns:
       Histogram of RGB G channel as a NumPy array image.
    """
    hist = np_rgb_channel_histogram(rgb, 1, "G")
    return hist


def np_rgb_b_histogram(rgb):
    """
    Obtain RGB B channel histogram as a NumPy array image.

    Args:
      rgb: Image as RGB NumPy array.

    Returns:
       Histogram of RGB B channel as a NumPy array image.
    """
    hist = np_rgb_channel_histogram(rgb, 2, "B")
    return hist


def pil_hue_histogram(h):
    """
    Create Matplotlib histogram of hue values for an HSV image and return the histogram as a PIL image.

    Args:
      h: Hue values as a 1-dimensional int NumPy array (scaled 0 to 360)

    Returns:
      Matplotlib histogram of hue values converted to a PIL image.
    """
    np_hist = np_hsv_hue_histogram(h)
    pil_hist = image_util.np_to_pil(np_hist)
    return pil_hist


def display_image_with_hsv_hue_histogram(np_rgb, text=None, scale_up=False):
    """
    Display an image with its corresponding hue histogram.

    Args:
      np_rgb: RGB image tile as a NumPy array
      text: Optional text to display above image
      scale_up: If True, scale up image to display by SCALE_FACTOR
    """
    hsv = filter.filter_rgb_to_hsv(np_rgb)
    h = filter.filter_hsv_to_h(hsv)
    np_hist = np_hsv_hue_histogram(h)
    hist_r, hist_c, _ = np_hist.shape

    if scale_up:
        np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=1)
        np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=0)

    img_r, img_c, img_ch = np_rgb.shape
    if text is not None:
        np_t = np_text(text)
        t_r, t_c, _ = np_t.shape
        t_i_c = max(t_c, img_c)
        t_i_r = t_r + img_r
        t_i = np.zeros([t_i_r, t_i_c, img_ch], dtype=np.uint8)
        t_i.fill(255)
        t_i[0:t_r, 0:t_c] = np_t
        t_i[t_r:t_r + img_r, 0:img_c] = np_rgb
        np_rgb = t_i  # for simplicity assign title+image to image
        img_r, img_c, img_ch = np_rgb.shape

    r = max(img_r, hist_r)
    c = img_c + hist_c
    combo = np.zeros([r, c, img_ch], dtype=np.uint8)
    combo.fill(255)
    combo[0:img_r, 0:img_c] = np_rgb
    combo[0:hist_r, img_c:c] = np_hist
    pil_combo = image_util.np_to_pil(combo)
    pil_combo.show()


def display_image(np_rgb, text=None, scale_up=False):
    """
    Display an image with optional text above image.

    Args:
      np_rgb: RGB image tile as a NumPy array
      text: Optional text to display above image
      scale_up: If True, scale up image to display by SCALE_FACTOR
    """
    if scale_up:
        np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=1)
        np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=0)

    img_r, img_c, img_ch = np_rgb.shape
    if text is not None:
        np_t = np_text(text)
        t_r, t_c, _ = np_t.shape
        t_i_c = max(t_c, img_c)
        t_i_r = t_r + img_r
        t_i = np.zeros([t_i_r, t_i_c, img_ch], dtype=np.uint8)
        t_i.fill(255)
        t_i[0:t_r, 0:t_c] = np_t
        t_i[t_r:t_r + img_r, 0:img_c] = np_rgb
        np_rgb = t_i

    pil_img = image_util.np_to_pil(np_rgb)
    pil_img.show()


def display_image_with_hsv_histograms(np_rgb, text=None, scale_up=False):
    """
    Display an image with its corresponding HSV hue, saturation, and value histograms.

    Args:
      np_rgb: RGB image tile as a NumPy array
      text: Optional text to display above image
      scale_up: If True, scale up image to display by SCALE_FACTOR
    """
    hsv = filter.filter_rgb_to_hsv(np_rgb)
    np_h = np_hsv_hue_histogram(filter.filter_hsv_to_h(hsv))
    np_s = np_hsv_saturation_histogram(filter.filter_hsv_to_s(hsv))
    np_v = np_hsv_value_histogram(filter.filter_hsv_to_v(hsv))
    h_r, h_c, _ = np_h.shape
    s_r, s_c, _ = np_s.shape
    v_r, v_c, _ = np_v.shape

    if scale_up:
        np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=1)
        np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=0)

    img_r, img_c, img_ch = np_rgb.shape
    if text is not None:
        np_t = np_text(text)
        t_r, t_c, _ = np_t.shape
        t_i_c = max(t_c, img_c)
        t_i_r = t_r + img_r
        t_i = np.zeros([t_i_r, t_i_c, img_ch], dtype=np.uint8)
        t_i.fill(255)
        t_i[0:t_r, 0:t_c] = np_t
        t_i[t_r:t_r + img_r, 0:img_c] = np_rgb
        np_rgb = t_i  # for simplicity assign title+image to image
        img_r, img_c, img_ch = np_rgb.shape

    hists_c = max(h_c, s_c, v_c)
    hists_r = h_r + s_r + v_r
    hists = np.zeros([hists_r, hists_c, img_ch], dtype=np.uint8)

    hists[0:h_r, 0:h_c] = np_h
    hists[h_r:h_r + s_r, 0:s_c] = np_s
    hists[h_r + s_r:h_r + s_r + v_r, 0:v_c] = np_v

    r = max(img_r, hists_r)
    c = img_c + hists_c
    combo = np.zeros([r, c, img_ch], dtype=np.uint8)
    combo.fill(255)
    combo[0:img_r, 0:img_c] = np_rgb
    combo[0:hists_r, img_c:c] = hists
    pil_combo = image_util.np_to_pil(combo)
    pil_combo.show()


def display_image_with_rgb_histograms(np_rgb, text=None, scale_up=False):
    """
    Display an image with its corresponding RGB histograms.

    Args:
      np_rgb: RGB image tile as a NumPy array
      text: Optional text to display above image
      scale_up: If True, scale up image to display by SCALE_FACTOR
    """
    np_r = np_rgb_r_histogram(np_rgb)
    np_g = np_rgb_g_histogram(np_rgb)
    np_b = np_rgb_b_histogram(np_rgb)
    r_r, r_c, _ = np_r.shape
    g_r, g_c, _ = np_g.shape
    b_r, b_c, _ = np_b.shape

    if scale_up:
        np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=1)
        np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=0)

    img_r, img_c, img_ch = np_rgb.shape
    if text is not None:
        np_t = np_text(text)
        t_r, t_c, _ = np_t.shape
        t_i_c = max(t_c, img_c)
        t_i_r = t_r + img_r
        t_i = np.zeros([t_i_r, t_i_c, img_ch], dtype=np.uint8)
        t_i.fill(255)
        t_i[0:t_r, 0:t_c] = np_t
        t_i[t_r:t_r + img_r, 0:img_c] = np_rgb
        np_rgb = t_i  # for simplicity assign title+image to image
        img_r, img_c, img_ch = np_rgb.shape

    hists_c = max(r_c, g_c, b_c)
    hists_r = r_r + g_r + b_r
    hists = np.zeros([hists_r, hists_c, img_ch], dtype=np.uint8)

    hists[0:r_r, 0:r_c] = np_r
    hists[r_r:r_r + g_r, 0:g_c] = np_g
    hists[r_r + g_r:r_r + g_r + b_r, 0:b_c] = np_b

    r = max(img_r, hists_r)
    c = img_c + hists_c
    combo = np.zeros([r, c, img_ch], dtype=np.uint8)
    combo.fill(255)
    combo[0:img_r, 0:img_c] = np_rgb
    combo[0:hists_r, img_c:c] = hists
    pil_combo = image_util.np_to_pil(combo)
    pil_combo.show()


def pil_text(text, w_border=TILE_TEXT_W_BORDER, h_border=TILE_TEXT_H_BORDER, font_path=FONT_PATH,
             font_size=TILE_TEXT_SIZE, text_color=TILE_TEXT_COLOR, background=TILE_TEXT_BACKGROUND_COLOR):
    """
    Obtain a PIL image representation of text.

    Args:
      text: The text to convert to an image.
      w_border: Tile text width border (left and right).
      h_border: Tile text height border (top and bottom).
      font_path: Path to font.
      font_size: Size of font.
      text_color: Tile text color.
      background: Tile background color.

    Returns:
      PIL image representing the text.
    """
    font = None
    if font_path is not None:
        font = ImageFont.truetype(font_path, font_size)
    x, y = ImageDraw.Draw(Image.new("RGB", (1, 1), background)).textsize(text, font)
    image = Image.new("RGB", (x + 2 * w_border, y + 2 * h_border), background)
    draw = ImageDraw.Draw(image)
    draw.text((w_border, h_border), text, text_color, font=font)
    return image


def np_text(text, w_border=TILE_TEXT_W_BORDER, h_border=TILE_TEXT_H_BORDER, font_path=FONT_PATH,
            font_size=TILE_TEXT_SIZE, text_color=TILE_TEXT_COLOR, background=TILE_TEXT_BACKGROUND_COLOR):
    """
    Obtain a NumPy array image representation of text.

    Args:
      text: The text to convert to an image.
      w_border: Tile text width border (left and right).
      h_border: Tile text height border (top and bottom).
      font_path: Path to font.
      font_size: Size of font.
      text_color: Tile text color.
      background: Tile background color.

    Returns:
      NumPy array representing the text.
    """
    pil_img = pil_text(text, w_border, h_border, font_path, font_size,
                       text_color, background)
    np_img = image_util.pil_to_np(pil_img)
    return np_img


def display_tile(tile, rgb_histograms=True, hsv_histograms=True):
    """
    Display a tile with its corresponding RGB and HSV histograms.

    Args:
      tile: The Tile object.
      rgb_histograms: If True, display RGB histograms.
      hsv_histograms: If True, display HSV histograms.
    """

    text = "S%03d R%03d C%03d\n" % (tile.slide_num, tile.r, tile.c)
    text += "Score:%4.2f Tissue:%5.2f%% CF:%2.0f SVF:%4.2f QF:%4.2f\n" % (
        tile.score, tile.tissue_percentage, tile.color_factor, tile.s_and_v_factor, tile.quantity_factor)
    text += "Rank #%d of %d" % (tile.rank, tile.tile_summary.num_tiles())

    np_scaled_tile = tile.get_np_scaled_tile()
    if np_scaled_tile is not None:
        small_text = text + "\n \nSmall Tile (%d x %d)" % (np_scaled_tile.shape[1], np_scaled_tile.shape[0])
        if rgb_histograms and hsv_histograms:
            display_image_with_rgb_and_hsv_histograms(np_scaled_tile, small_text, scale_up=True)
        elif rgb_histograms:
            display_image_with_rgb_histograms(np_scaled_tile, small_text, scale_up=True)
        elif hsv_histograms:
            display_image_with_hsv_histograms(np_scaled_tile, small_text, scale_up=True)
        else:
            display_image(np_scaled_tile, small_text, scale_up=True)

    np_tile = tile.get_np_tile()
    text += " based on small tile\n \nLarge Tile (%d x %d)" % (np_tile.shape[1], np_tile.shape[0])
    if rgb_histograms and hsv_histograms:
        display_image_with_rgb_and_hsv_histograms(np_tile, text)
    elif rgb_histograms:
        display_image_with_rgb_histograms(np_tile, text)
    elif hsv_histograms:
        display_image_with_hsv_histograms(np_tile, text)
    else:
        display_image(np_tile, text)


def display_image_with_rgb_and_hsv_histograms(np_rgb, text=None, scale_up=False):
    """
    Display a tile with its corresponding RGB and HSV histograms.

    Args:
      np_rgb: RGB image tile as a NumPy array
      text: Optional text to display above image
      scale_up: If True, scale up image to display by SCALE_FACTOR
    """
    hsv = filter.filter_rgb_to_hsv(np_rgb)
    np_r = np_rgb_r_histogram(np_rgb)
    np_g = np_rgb_g_histogram(np_rgb)
    np_b = np_rgb_b_histogram(np_rgb)
    np_h = np_hsv_hue_histogram(filter.filter_hsv_to_h(hsv))
    np_s = np_hsv_saturation_histogram(filter.filter_hsv_to_s(hsv))
    np_v = np_hsv_value_histogram(filter.filter_hsv_to_v(hsv))

    r_r, r_c, _ = np_r.shape
    g_r, g_c, _ = np_g.shape
    b_r, b_c, _ = np_b.shape
    h_r, h_c, _ = np_h.shape
    s_r, s_c, _ = np_s.shape
    v_r, v_c, _ = np_v.shape

    if scale_up:
        np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=1)
        np_rgb = np.repeat(np_rgb, SCALE_FACTOR, axis=0)

    img_r, img_c, img_ch = np_rgb.shape
    if text is not None:
        np_t = np_text(text)
        t_r, t_c, _ = np_t.shape
        t_i_c = max(t_c, img_c)
        t_i_r = t_r + img_r
        t_i = np.zeros([t_i_r, t_i_c, img_ch], dtype=np.uint8)
        t_i.fill(255)
        t_i[0:t_r, 0:t_c] = np_t
        t_i[t_r:t_r + img_r, 0:img_c] = np_rgb
        np_rgb = t_i  # for simplicity assign title+image to image
        img_r, img_c, img_ch = np_rgb.shape

    rgb_hists_c = max(r_c, g_c, b_c)
    rgb_hists_r = r_r + g_r + b_r
    rgb_hists = np.zeros([rgb_hists_r, rgb_hists_c, img_ch], dtype=np.uint8)
    rgb_hists[0:r_r, 0:r_c] = np_r
    rgb_hists[r_r:r_r + g_r, 0:g_c] = np_g
    rgb_hists[r_r + g_r:r_r + g_r + b_r, 0:b_c] = np_b

    hsv_hists_c = max(h_c, s_c, v_c)
    hsv_hists_r = h_r + s_r + v_r
    hsv_hists = np.zeros([hsv_hists_r, hsv_hists_c, img_ch], dtype=np.uint8)
    hsv_hists[0:h_r, 0:h_c] = np_h
    hsv_hists[h_r:h_r + s_r, 0:s_c] = np_s
    hsv_hists[h_r + s_r:h_r + s_r + v_r, 0:v_c] = np_v

    r = max(img_r, rgb_hists_r, hsv_hists_r)
    c = img_c + rgb_hists_c + hsv_hists_c
    combo = np.zeros([r, c, img_ch], dtype=np.uint8)
    combo.fill(255)
    combo[0:img_r, 0:img_c] = np_rgb
    combo[0:rgb_hists_r, img_c:img_c + rgb_hists_c] = rgb_hists
    combo[0:hsv_hists_r, img_c + rgb_hists_c:c] = hsv_hists
    pil_combo = image_util.np_to_pil(combo)
    pil_combo.show()
