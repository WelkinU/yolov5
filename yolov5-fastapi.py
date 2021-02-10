from fastapi import FastAPI, Request, BackgroundTasks, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
#from pydantic import BaseModel #not actually using pydantic validation right now

from typing import Optional, List
from starlette.responses import RedirectResponse, FileResponse
import starlette.status as status

import subprocess
import os
from PIL import Image
from io import BytesIO
import uuid

import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory = 'templates')

#so we can read main.css
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

os.makedirs('./data/fastapi_download', exist_ok = True)

@app.get("/")
def home(request: Request):
	'''
	Displays home page
	'''

	return templates.TemplateResponse('home.html', {
			"request": request,
		})

@app.post("/")
async def process_home_form(file: UploadFile = File(...), 
					model_name: str = Form(...),
					response_method: str = Form(...)):
	
	'''
	Requires an image file upload, model name (ex. yolov5s.pt) and response method ("view" or "download").

	Response method "download" returns a txt file output of detect.py (normalized xywh bboxes).
	Response method "view" is intended for the webapp - returns html page with output image / text data in table format.

	This function does the following:
	1. Receives form data from home page / API request
	2. Saves uploaded file to /data/fastapi_download
	3. Runs YOLOv5 model on the data
	4. Redirects to /inference to view results, or to /download to download results
	'''

	tag = str(uuid.uuid4()) #create a random tag for this newly uploaded file
	image_path = f'./data/fastapi_download/{tag}.png'

	image = Image.open(BytesIO(await file.read()))
	image.save(image_path)
	
	''' 
	#use this to upload any file type, not just images
	with tempfile.NamedTemporaryFile() as temp:
		tag = os.path.basename(temp.name)

		#saving UploadFile to NamedTemporaryFile, 
		#not sure if there's more efficent way to do this without modifying YOLO dataloader
		content = await file.read() #async read
		await file.write(content) #async write to tempfile
	'''

	#run detect.py via command line to minimize changes to detect.py
	#not currently run as a background task
	cmd = " ".join(["python detect.py",
				"--weights",model_name,
				"--source",image_path,
				"--device", "0",
				"--agnostic-nms", "--save-txt",
				"--fastapi-outtag", tag,
				])
	subprocess.call(cmd, shell = True)

	if response_method == 'view':
		#if user selected "View Results In Broswer"
		url = '/inference?tag='+tag
	
	else: # if response method was "download" (or anything other than "view")

		#not sure if this is too fancy, could just put the Redirect response right here
		#this lets the user go to the same download url later, and not have to reupload the file
		url = '/download?tag='+tag
	
	return RedirectResponse(url=url, status_code=status.HTTP_302_FOUND)

@app.get("/inference/")
def inference(request: Request, tag: Optional[str] = None):
	'''
	Display a YOLO result image with bboxes drawn and table of data
	'''
	txt_file = f'./data/fastapi_download/{tag}.txt'
	df = pd.read_csv(txt_file, delimiter = ' ', names = ['cls','x','y','w','h'])

	return templates.TemplateResponse('inference.html', 
			{"request": request,
			'tag': tag,
			'df': df,
			})


@app.get("/download/")
def download_results(request: Request, tag: Optional[str] = None):
	'''
	Downloads YOLO results txt file containing bbox info in normalized xywh format
	'''
	#only return the txt file for now
	#to return bbox image + txt file: https://stackoverflow.com/questions/61163024/return-multiple-files-from-fastapi

	#make sure file is downloaded as attachment with the proper filename
	headers = {"Content-Disposition": f'attachment; filename="{tag}.txt"'}
	
	return FileResponse(f'./data/fastapi_download/{tag}.txt',
					headers = headers)


@app.get("/image/")
def get_image(request: Request, tag: Optional[str] = None):
	'''
	Get a YOLO result image with bboxes drawn, helper function for the /inference image display
	'''

	image_path = os.path.realpath(f'./data/fastapi_download/{tag}.png')
	return FileResponse(image_path)

@app.get("/about/")
def about_us(request: Request):
	'''
	Display about us page
	'''

	return templates.TemplateResponse('about.html', 
			{"request": request})
