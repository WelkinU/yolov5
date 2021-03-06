from nuimages import NuImages
from tqdm import tqdm
import os
import yaml

#---------------------------------- DEFINE CONSTANTS HERE -----------------------------------------
DATA_ROOT = os.path.dirname(os.path.realpath(__file__))

#nuimages has same image dimensions for all images
IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 900

'''class_map dict maps nuimages classes to other class names.
ex. human.pedestrian.adult and human.pedestrian.child both get mapped to "person"
vehicle.ego is removed from the data labels because it's string is "None"
'''
class_map = {
			'animal': 'animal',
			'human.pedestrian.adult': 'person',
			'human.pedestrian.child': 'person',
			'human.pedestrian.construction_worker': 'person',
			'human.pedestrian.personal_mobility': 'person',
			'human.pedestrian.police_officer': 'person',
			'human.pedestrian.stroller': 'person',
			'human.pedestrian.wheelchair': 'person',
			'movable_object.barrier': 'barrier',
			'movable_object.debris': 'debris',
			'movable_object.pushable_pullable': 'object_pushable_pullable',
			'movable_object.trafficcone': 'traffic cone',
			'static_object.bicycle_rack': 'bicycle rack',
			'vehicle.bicycle': 'bicycle',
			'vehicle.bus.bendy': 'bus',
			'vehicle.bus.rigid': 'bus',
			'vehicle.car': 'car',
			'vehicle.construction': 'construction vechicle',
			'vehicle.emergency.ambulance': 'truck',
			'vehicle.emergency.police': 'car',
			'vehicle.motorcycle': 'motorcycle',
			'vehicle.trailer': 'trailer',
			'vehicle.truck': 'truck',
			'vehicle.ego': 'None', #ignore ego vehicle
			}

#-------------------------------  AutoGenerate nuimages.yaml file -------------------------------------
#Create YAML file per section #1 of https://github.com/ultralytics/yolov5/issues/12
with open(os.path.join(DATA_ROOT,'nuimages.yaml'),'w') as file:
	classes = sorted(list(set(class_map.values())))
	classes = [c for c in classes if c!= 'None']
	print(classes)
	
	data = {'train': os.path.join(DATA_ROOT,'images','train'),
			'val': os.path.join(DATA_ROOT,'images','val'),
			'nc': len(classes),
			'names': classes,
			}

	yaml.dump(data, file, sort_keys = False)

#-------------------------------  Dump Images & Create Label files -------------------------------------
#Create files per section #2 of https://github.com/ultralytics/yolov5/issues/12

for version in ['train','val']:
	print(f'Processing {version} set...')

	#make image and label output directories if they don't exist
	os.makedirs(os.path.join(DATA_ROOT,'images',version),exist_ok = True)
	os.makedirs(os.path.join(DATA_ROOT,'labels',version),exist_ok = True)

	nuim = NuImages(dataroot=DATA_ROOT, version='v1.0-'+version, verbose=True, lazy=True)

	#create reverse class index dictionary: {animal -> 0, barrier -> 1, bicycle -> 2, ...}
	class_index_map = {obj_class:classes.index(class_map[obj_class]) for obj_class in class_map if class_map[obj_class] != 'None'}

	for sample_idx in tqdm(range(0,len(nuim.sample))):
		sample = nuim.get('sample', nuim.sample[sample_idx]['token'])
		key_camera_token = sample['key_camera_token']

		nuim.render_image(key_camera_token, annotation_type='none',with_category=True, with_attributes=True, box_line_width=-1, render_scale=5, 
							out_path = os.path.join(DATA_ROOT,'images',version,f'{sample_idx}.jpg'))		

		object_tokens, surface_tokens = nuim.list_anns(sample['token'], verbose = False)

		with open(os.path.join(DATA_ROOT,'labels',version,f'{sample_idx}.txt'),'w') as file:
			for object_token in object_tokens:
				token_data = nuim.get('object_ann',object_token)
				token_name = nuim.get('category',token_data['category_token'])['name']

				if class_map[token_name] != 'None':
					file.writelines('{} {} {} {} {}\n'.format(class_index_map[token_name],
															(token_data['bbox'][2] + token_data['bbox'][0])/(IMAGE_WIDTH * 2),
															(token_data['bbox'][3] + token_data['bbox'][1])/(IMAGE_HEIGHT * 2),
															(token_data['bbox'][2] - token_data['bbox'][0])/IMAGE_WIDTH,
															(token_data['bbox'][3] - token_data['bbox'][1])/IMAGE_HEIGHT,
															))
