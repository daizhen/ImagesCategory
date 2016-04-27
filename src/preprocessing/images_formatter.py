from PIL import Image
import os
import sys
def convert_to_jpg(in_dir,out_dir):
	listfile=os.listdir(in_dir)
	for current_file in listfile:
		in_full_name = os.path.join(in_dir,current_file)
		outfile = os.path.splitext(current_file)[0] + ".jpeg"
		out_full_name = os.path.join(out_dir,outfile)
		if not os.path.exists(out_full_name):
			if current_file.lower().endswith('.png'):
				try:
					Image.open(in_full_name).save(out_full_name)
				except IOError:
					print "This format can not support!", current_file
					open(out_full_name, "wb").write(open(in_full_name, "rb").read())
			elif current_file.lower().endswith('.jpg'):
				#Copy file
				open(out_full_name, "wb").write(open(in_full_name, "rb").read())
			elif current_file.lower().endswith('.jpeg'):
				open(out_full_name, "wb").write(open(in_full_name, "rb").read())
			elif current_file.lower().endswith('.gif'):
				open(out_full_name, "wb").write(open(in_full_name, "rb").read())
			else:
				print current_file
convert_to_jpg('../data/original_images','../data/jpg_images')