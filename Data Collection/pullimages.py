import fitz, io, sys, os
#import PySimpleGUI as sg
from PIL import Image
from random import random
dir_path = os.path.dirname(os.path.realpath(__file__))

# Function to extract images from PDF files in a folder
def pull_all_images():
    save_dir = [dir_path + "/data/train/Not",dir_path + "/data/test/Not"]

    if False:
        # Check if a folder path is provided as a command-line argument, otherwise prompt the user
        fname = sys.argv[1] if len(sys.argv) == 2 else None
        if not fname:
            fname = sg.PopupGetFolder("Select folder:", title="PyMuPDF PDF Image Extraction")
        if not fname:
            raise SystemExit()
    else:
        fname = dir_path + "/downloaded pdfs"

    total_images = 0

    # Iterate through the files in the selected folder
    for file in os.listdir(fname):
        file_path = os.path.join(fname, file)

        # Check if the file is not empty
        if os.path.getsize(file_path) > 0:
            total_images += extract_images_from_pdf(file_path, save_dir)

    print(f"Total images extracted: {total_images}")

# Function to extract images from a PDF file and save them to a directory
def extract_images_from_pdf(pdf_path, save_dir):
    pdf_file = fitz.open(pdf_path)
    total_images = 0

    # Iterate through the pages of the PDF
    for page_index in range(len(pdf_file)):
        page = pdf_file[page_index]
        image_list = page.get_images()

        if image_list:
            print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
        else:
            print("[!] No images found on page", page_index)

        # Iterate through the images on the page
        for image_index, img in enumerate(page.get_images(), start=1):
            xref = img[0]
            base_image = pdf_file.extract_image(xref)

            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image = Image.open(io.BytesIO(image_bytes))
            if image.size[0] > 200 and image.size[1] > 200:
                image_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}-{page_index+1}-{image_index}.{image_ext}"
                if False:
                    save_path = dir_path + "/to be filtered"
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    image_path = os.path.join(save_path, image_filename)
                elif random() > 0.2:
                    image_path = os.path.join(save_dir[0], image_filename)
                else:
                    image_path = os.path.join(save_dir[1], image_filename)
                image.save(image_path)
                total_images += 1

    return total_images

# Call the main function to extract images from PDF files
pull_all_images()
