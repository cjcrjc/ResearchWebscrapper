import fitz
import io
from PIL import Image
import os
import sys
import PySimpleGUI as sg

save_dir = "data"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

fname = sys.argv[1] if len(sys.argv) == 2 else None
if not fname:
    fname = sg.PopupGetFolder("Select folder:", title="PyMuPDF PDF Image Extraction")
if not fname:
    raise SystemExit()

i=0
for file in os.listdir(fname):
	if os.path.getsize(fname + "/" + file) > 0:
		i+=1
		pdf_file = fitz.open(fname + "/" + file)

		for page_index in range(len(pdf_file)):
			page = pdf_file[page_index]
			image_list = page.get_images()

			if image_list:
				print(
					f"[+] Found a total of {len(image_list)} images in page {page_index}")
			else:
				print("[!] No images found on page", page_index)
			for image_index, img in enumerate(page.get_images(), start=1):
				#print(image_index)
				xref = img[0]
				base_image = pdf_file.extract_image(xref)
				
				image_bytes = base_image["image"]
				image_ext = base_image["ext"]
				
				image = Image.open(io.BytesIO(image_bytes))
				image.save(f"{save_dir}/{file[0:5]}-{page_index+1}-{image_index}.{image_ext}")
print(i)