#! /usr/bin/env python

# a tool that chops a photo into like-pixel groups
# @author: patrick kage

# this program makes some assumptions about the input file
# 1. the text is arranged in rows
# 2. the text is in a single column layout

from scipy import misc
import numpy as np
import os, uuid, time
from multiprocessing import Pool, cpu_count
from functools import partial
from copy import deepcopy

class Photochopper:
	def __init__(self, orig_image, threshold):
		# load the image in from a file
		self.original = misc.imread(orig_image, flatten=True);
		print("original shape = " + str(self.original.shape));

		# create an array of zeroes to store which pixels we've already hit 
		self.hits = np.zeros(self.original.shape);

		# create an array that'll hold the groups we've already identified
		self.groups = [];

		# set the threshold
		self.threshold = threshold;

		# set the threadcount
		try:
			self.threadcount = cpu_count();
		except NotImplementedError:
			self.threadcount = 8;
			print('cpu_count() is not implemented on this system.');

	def set_max_threads(self, threads):
		self.threadcount = threads;

	def process(self):
		
		# store all the vertical ranges that have text on them
		rows = [];
		current = [0,0];
		active = False;



		# do a first pass along the image to find ranges of row with stuff on them
		for y in range(0, self.original.shape[0]):
			
			thisrow = False;
			for x in range(0, self.original.shape[1]):
				if self.original[y][x] < self.threshold:
					thisrow = True;
					break;
			#print('row ' + str(y) + ' is ' + ('empty' if not thisrow else 'not empty'));
			if thisrow:
				if not active:
					active = True;
					current[0] = y;
			else:
				if active:
					active = False;
					current[1] = y;
					print('found a range from Y:' + str(current[0]) + ' to Y:' + str(current[1]));
					rows.append(deepcopy(current));


		# commence shoehorned multithreading
		print('slicing data for multithreading...');
		slices = [];
		for row in rows:
			slices.append(self.original[row[0]:row[1]]);

		# create the thread pool
		threads = Pool(self.threadcount);
		print("created pool of up to " + str(self.threadcount) + " threads");

		# create a partial
		processor = partial(process_row, self.threshold);
		tmpgroups = threads.map(processor, slices);

		for x in tmpgroups:
			self.groups += x;

	def export_groups(self, dir_name):
		if not os.path.exists('out'):
			os.mkdir('out');
		os.mkdir('out/' + dir_name);
		print('created directory out/' + dir_name);
		
		i = 0;
		for group in self.groups:
			misc.imsave('out/' + dir_name + '/' + str(i) + '.png', group);
			i += 1;
		print('saved all groups');

# this is because of multithreading
def process_row(threshold, original):
	groups = [];
	final = [];
	hits = np.zeros(original.shape);

	# look for connected pixels
	for x in range(0, original.shape[1]):
		for y in range(0, original.shape[0]):
			if original[y][x] < threshold and hits[y][x] == 0:
				# (re)create a sparse array
				group = _SparseArray();
				# make sure we look around the origin point
				group.set(y,x,-1);

				# this should be better but it isn't
				while True:
					# pixels to look at next
					pxls = group.get_all_of(-1);
					for pixel in pxls:
						# there should be a better way of doing this
						orig = original[y][x];
						group.set(pixel.y, pixel.x, orig);

						# search around the pixel
						for sy in range(-1, 2):
							for sx in range(-1, 2):
								if abs(sx) + abs(sy) == 2:
									continue;
								try:
									if original[pixel.y + sy][pixel.x + sx] < threshold and group.get(pixel.y + sy, pixel.x + sx) == None and hits[pixel.y + sy][pixel.x + sx] == 0:
										group.set(pixel.y + sy, pixel.x + sx, -1);
								except:
									pass; # shhhhh... it's okay.

					if not group.contains(-1):
						break;

				# set the pixels we've hit into the hit group
				for pxl in group.arr:
					hits[pxl.y][pxl.x] = 1;

				points = group.get_bounding_points();
				# brag a little bit
				print('identified a group of ' + str(len(group.arr)) + ' pixels spanning from ' + str(points[0]) + ' to ' + str(points[1]));
				groups.append(deepcopy(group));

	# align diacritics
	addedMultipart = False;
	for i in range(0, len(groups) - 1):
		if addedMultipart:
			addedMultipart = False;
			continue;
		
		# TODO: make this combine diacretics as well
		if groups[i].get_shape()[1] == groups[i + 1].get_shape()[1]:
			group = _SparseArray();
			group.integrate(groups[i]);
			group.integrate(groups[i + 1]);
			final.append(group.export());
			addedMultipart = True;
		else:
			final.append(groups[i].export());

	print('thread finished');
	return final;


# Class to hold a pixel value
class _Pixel():
	def __init__(self, y, x, val):
		self.x = x;
		self.y = y;
		self.val = val;

# sparse array
class _SparseArray():
	def __init__(self):
		self.arr = [];

	# gets a specific pixel
	def get(self, y, x):
		for pixel in self.arr:
			if pixel.y == y and pixel.x == x:
				return pixel.val;
		return None;

	# sets a pixel
	def set(self, y, x, val):
		for i in range(0, len(self.arr)):
			pixel = self.arr[i];
			if pixel.y == y and pixel.x == x:
				self.arr[i] = _Pixel(y, x, val);
		self.arr.append(_Pixel(y, x, val));
		return None;

	# gets a list of all pixels with a specific value
	def get_all_of(self, val):
		out = [];
		for pixel in self.arr:
			if pixel.val == val:
				out.append(pixel);
		return out;

	# check if the array contains a value
	def contains(self, val):
		out = False;
		for pixel in self.arr:
			if pixel.val == val:
				out = True;
		return out;

	# gets the min y, min x, max y, and max x of the array
	def get_shape(self):
		lowest_x = self.arr[0].x;
		highest_x = self.arr[0].x;
		lowest_y = self.arr[0].y;
		highest_y = self.arr[0].y;

		for pixel in self.arr:
			if pixel.x < lowest_x:
				lowest_x = pixel.x;
			if pixel.x > highest_x:
				highest_x = pixel.x;
			if pixel.y < lowest_y:
				lowest_y = pixel.y;
			if pixel.y > highest_y:
				highest_y = pixel.y;

		return (lowest_y, lowest_x, highest_y, highest_x);

	# converts the shape to a pair of points
	def get_bounding_points(self):
		shape = self.get_shape();
		return ((shape[0], shape[1]), (shape[0], shape[1]));

	# exports to a numpy array
	def export(self):
		shape = self.get_shape();
		height = shape[2] - shape[0];
		width = shape[3] - shape[1];

		export = np.zeros((height + 1, width + 1));
		export.fill(255);

		for pixel in self.arr:
			export[pixel.y - shape[0]][pixel.x - shape[1]] = pixel.val;

		return export;

	# assimilate another sparse array into itself
	def integrate(self, group):
		for pixel in group.arr:
			self.set(pixel.y, pixel.x, pixel.val);

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="dice an image, separating out groups of dark pixels");
	parser.add_argument('filename', type=str, help="an input png file");
	parser.add_argument('--process-count', type=int, help="number of worker processes to use. uses the cpu core count by default");
	opts = parser.parse_args(); 

	dicer = Photochopper(opts.filename, 150);
	start_time = time.clock();
	dicer.process();
	end_time = time.clock();
	print('processing took ' + str(end_time - start_time) + ' seconds.');

	start_time = time.clock();
	dicer.export_groups(str(uuid.uuid4()));
	end_time = time.clock();
	print('saving took ' + str(end_time - start_time) + ' seconds.');
	
