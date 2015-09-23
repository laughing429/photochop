#! /usr/bin/env python

# a tool that chops a photo into like-pixel groups
# @author: patrick kage

from scipy import misc
import numpy as np
import os, uuid, time
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

		i = 0
		for rng in rows:
			print('processing ' + str(rng));
			for x in range(0, self.original.shape[1]):
				for y in range(rng[0], rng[1]):
					if self.original[y][x] < self.threshold and self.hits[y][x] == 0:
						self.groups.append(self.__get_connected_pixels(y,x));
						i += 1
		print('total matches: ' + str(i) + ', total ranges: ' + str(len(rows)));

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

	def __get_connected_pixels(self, y, x):
		# first, create a sparse array to hold the group
		group = _SparseArray();

		# create a pixel with a value of -1 (means we should paint around it)
		group.set(y, x, -1);

		# grab all connected pixels
		while True:
			# pixels to look at next 
			pxls = group.get_all_of(-1);
			for pixel in pxls:
				orig = self.original[y][x];
				group.set(pixel.y, pixel.x, orig);
				#print('searching for ' + str(pixel.y) + ', ' + str(pixel.x) + ' : ' + str(pixel.val) );

				for sy in range(-1, 2):
					for sx in range(-1, 2):
						if self.original[pixel.y + sy][pixel.x + sx] < self.threshold and group.get(pixel.y + sy, pixel.x + sx) == None and self.hits[pixel.y + sy][pixel.x + sx] == 0:
							group.set(pixel.y + sy, pixel.x + sx, -1);

			if not group.contains(-1):
				break; 

		# mark the pixels we've found as hit
		for pxl in group.arr:
			self.hits[pxl.y][pxl.x] = 1;

		points = group.get_bounding_points();
		print('identified a group of ' + str(len(group.arr)) + ' pixels spanning from ' + str(points[0]) + ' to ' + str(points[1]));
		return group.export();




class _Pixel():
	def __init__(self, y, x, val):
		self.x = x;
		self.y = y;
		self.val = val;

class _SparseArray():
	def __init__(self):
		self.arr = [];

	def get(self, y, x):
		for pixel in self.arr:
			if pixel.y == y and pixel.x == x:
				return pixel.val;
		return None;

	def set(self, y, x, val):
		for i in range(0, len(self.arr)):
			pixel = self.arr[i];
			if pixel.y == y and pixel.x == x:
				self.arr[i] = _Pixel(y, x, val);
		self.arr.append(_Pixel(y, x, val));
		return None;

	def get_all_of(self, val):
		out = [];
		for pixel in self.arr:
			if pixel.val == val:
				out.append(pixel);
		return out;

	def contains(self, val):
		out = False;
		for pixel in self.arr:
			if pixel.val == val:
				out = True;
		return out;

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

	def get_bounding_points(self):
		shape = self.get_shape();
		return ((shape[0], shape[1]), (shape[0], shape[1]));

	def export(self):
		shape = self.get_shape();
		height = shape[2] - shape[0];
		width = shape[3] - shape[1];

		export = np.zeros((height + 1, width + 1));
		export.fill(255);

		for pixel in self.arr:
			export[pixel.y - shape[0]][pixel.x - shape[1]] = pixel.val;

		return export;



if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="dice an image, separating out groups of dark pixels");
	parser.add_argument('filename', type=str, help="an input png file");
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
	

