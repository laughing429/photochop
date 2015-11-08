#! /usr/bin/env python

import os, json, sys
from unidecode import unidecode

idir = sys.argv[1];
odir = sys.argv[2];

os.system('mkdir ' + odir);

for case in os.listdir(idir):
	if case == '.DS_Store':
		continue;
	for tag in os.listdir(idir + '/' + case):
		if tag == '.DS_Store':
			continue;
		for f in os.listdir(idir + '/' + case + '/' + tag):
			if f == '.DS_Store':
				continue;
			print(f + '\t' + case + '\t' + tag[-1])
			os.system('cp "%s/%s/%s/%s" "%s/%s"' % (idir, case, tag, f, odir, f));

