#! /usr/bin/env python

import os, shutil, argparse

parser = argparse.ArgumentParser();
parser.add_argument('--tags');
parser.add_argument('--indir');
parser.add_argument('--outdir');
parser.add_argument('--ignore');
args = parser.parse_args();


tagfile = open(args.tags, 'r').read();
ignorelist = open(args.ignore, 'r').readlines();
ignorelist = [i.strip() for i in ignorelist];


indir = args.indir;
outdir = args.outdir;

print(ignorelist);

os.mkdir(outdir);

i = 0;
for f in os.listdir(indir):
#	nextc = tagfile[i] + (tagfile[i + 1] if i < len(tagfile) else '\n');
	if not f in ignorelist:
		shutil.copy2(indir + '/' + f, outdir + '/' + (tagfile[i] + '.').join(f.split('.')));
	else:
		i += 1;
		print('skipped ' + f);
	i += 1;

