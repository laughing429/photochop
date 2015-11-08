#! /usr/bin/env python

import os, sys


# get the in directory
idir = sys.argv[1];

# create the output directory if it doesnt exist
if not os.path.exists('out'):
	os.system('mkdir out')

for f in os.listdir(idir):
	print('handling ' + f);
	os.system('cp %s/%s %s' % (idir, f, f));
	os.system('tesseract %s ocr -psm 10' % f);
	ret = open('ocr.txt', 'r').read().strip();
	case = 'lower' if ret.lower() == ret else 'upper';
	print('\t\t\tread as "%s" (%s case)' % (ret, case));
	odir = 'out/%s/tag_%s/' % (case, ret);

	os.system('mkdir -p %s; mv %s %s' % (odir, f, odir));
	print('mkdir -p %s; mv %s %s' % (odir, f, odir));
