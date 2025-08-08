import glob
from PIL import Image, ImageDraw
import argparse
import os

def leading_num(x):
    return int(x.split("/")[-1].split("_")[0])

def make_gif(renders_glob, output_path, duration=30, background="WHITE"):
    renders = glob.glob(renders_glob)
    if len(renders) == 0:
        return False
    image_paths = sorted(glob.glob(renders_glob))
    imgs = [Image.open(f) for f in image_paths]

    if background and imgs[0].mode == "RGBA":
        new_imgs = []
        for img in imgs:
            new_image = Image.new("RGBA", img.size, background) # Create a color rgba background
            new_image.paste(img, (0, 0), img) # Paste the image on the background.
            new_imgs.append(new_image.convert("RGB"))
        close_all(imgs)
        imgs = new_imgs

    stack_gif(output_path, imgs, duration=duration)
    close_all(imgs)
    return True

def stack_gif(output_path, imgs, duration=30):
    imgs[0].save(fp=output_path, format='GIF', append_images=imgs[1:],
        save_all=True, duration=duration, loop=0, disposal=0)

def close_all(imglist):
    for img in imglist:
        img.close()

def compare_imgs(imgs, names=None, resize=False):
    width, height = imgs[0].size # expect images to be the same
    header_height = 20 if names is not None else 0
    result_img = Image.new('RGB', (width*len(imgs), height+header_height))
    draw = ImageDraw.Draw(result_img)
    draw.rectangle([(0, 0), (width*len(imgs), height+header_height)], fill="white")
    for i, img in enumerate(imgs):
        if resize:
            img = img.resize((width, height))
        result_img.paste(img, (i*width, header_height))
        if names is not None:
            draw.text((int((i+0.5)*width), header_height//2), names[i], fill="black", anchor="mm")
    return result_img

def compare_grid(griddir, colnames=None, rownames=None, gtdir=None, gtlabel="GT", init_idx=0):
    """ Combine images in a grid for gif making
    griddir: list of lists of image paths (inner list is the row, outer list is the column)

    """
    from PIL import Image, ImageDraw, ImageFont
    import cv2

    width, height = 224, 224
    top_header = 20 if colnames is not None else 0
    left_header = 200 if rownames is not None else 0

    # Check that all rows are the same length
    assert len(set(len(row) for row in griddir)) == 1, "All rows must have the same number of images"

    if rownames is not None:
        assert len(rownames) == len(griddir), "Number of row names must match number of rows"

    if colnames is not None:
        assert len(colnames) == len(griddir[0]), "Number of column names must match number of columns"

    nrows = len(griddir)
    ncols = len(griddir[0])

    # Get the longest image directory set and mod the rest to match
    gridlengths = []
    for x, row in enumerate(griddir):
        gridlength = []
        for y, imgpath in enumerate(row):
            gridlength.append(len(glob.glob(os.path.join(imgpath, "*.png"))))
        gridlengths.append(gridlength)

    gtlengths = None
    # NOTE: gtdir should be list of length 1 or length of the # rows
    if gtdir is not None:
        if len(gtdir) == 1:
            # Get GT lengths
            gtlengths = [len(glob.glob(os.path.join(gtdir[0], "*.png")))] * len(griddir)
        else:
            assert len(gtdir) == len(griddir), f"gtdir should be length 1 or same as # rows!"
            gtlengths = [len(glob.glob(os.path.join(gt, "*.png"))) for gt in gtdir]

        max_length = max([max(row) for row in gridlengths] + gtlengths)
        total_width, total_height = width*(ncols + 1)+left_header, height*nrows+top_header
    else:
        max_length = max([max(row) for row in gridlengths])
        total_width, total_height = width*ncols+left_header, height*nrows+top_header

    imgs = []
    for i in range(max_length):
        img = Image.new('RGB', (total_width, total_height))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (total_width, total_height)], fill="white")
        font = ImageFont.truetype(os.path.join(cv2.__path__[0],'qt','fonts', 'DejaVuSans.ttf'), 20)

        for x, row in enumerate(griddir):
            # Paste the row images
            # GT image
            buffer = 0

            if gtlengths is not None:
                buffer = 1
                if gtlengths[x] > 0:
                    gtidx = i % gtlengths[x]

                    if len(gtdir) == 1:
                        gt = gtdir[0]
                    else:
                        gt = gtdir[x]

                    try:
                        with Image.open(os.path.join(gt, f"{gtidx + init_idx:03}.png")) as subimg:
                            subimg = subimg.resize((width, height))
                            img.paste(subimg, (left_header, x*height+top_header))
                    except:
                        with Image.open(os.path.join(gt, f"{gtidx + init_idx:04}.png")) as subimg:
                            subimg = subimg.resize((width, height))
                            img.paste(subimg, (left_header, x*height+top_header))

            # NOTE: Last image in the row is the GT
            for y, imgpath in enumerate(row):

                if gridlengths[x][y] == 0:
                    continue

                imgidx = i % gridlengths[x][y]
                try:
                    with Image.open(os.path.join(imgpath, f"{imgidx + init_idx:03}.png")) as subimg:
                        subimg = subimg.resize((width, height), Image.BICUBIC)
                        img.paste(subimg, ((y + buffer)*width+left_header, x*height+top_header))
                except:
                    with Image.open(os.path.join(imgpath, f"{imgidx + init_idx:04}.png")) as subimg:
                        subimg = subimg.resize((width, height), Image.BICUBIC)
                        img.paste(subimg, ((y + buffer)*width+left_header, x*height+top_header))

            # Write the row labels
            if rownames is not None:
                draw.text((left_header//2, x*height + height // 2 +top_header), rownames[x], fill="black", anchor="mm", font=font)

        # GT label
        if gtlengths is not None:
            draw.text((int(0.5*width + left_header), top_header//2), gtlabel, fill="black", anchor="mm", font=font)

        # Write the column labels
        for y, viewname in enumerate(colnames):
            draw.text((int((y+0.5 + buffer)*width + left_header), top_header//2), viewname, fill="black", anchor="mm", font=font)

        imgs.append(img)

    return imgs

if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", type=json.loads)
    parser.add_argument("--output", required=True, help="output file path")
    parser.add_argument("--colnames", nargs="*", type=str, help="column labels")
    parser.add_argument("--rownames", nargs="*", type=str, help="row labels")
    parser.add_argument("--gtdir", nargs="*", type=str, help="list of paths to gt images dir")
    parser.add_argument("--gtlabel", default="GT", type=str, help="label of gt column")
    parser.add_argument("--resize", action="store_true", help="resize all images to match first")
    parser.add_argument("--init_idx", type=int, default=0)
    args = parser.parse_args()

    # imgs = [[Image.open(f) for f in sorted(glob.glob(os.path.join(target, "*.png")))] for target in args.targets]
    # imgs = list(zip(*imgs))
    # output_imgs = [compare_imgs(img_row, args.names) for img_row in imgs]

    output_imgs = compare_grid(args.targets, args.colnames, args.rownames, args.gtdir,
                               gtlabel=args.gtlabel, init_idx=args.init_idx)
    outputdir = os.path.dirname(args.output)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    stack_gif(args.output, output_imgs)
