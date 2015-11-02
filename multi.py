#! /usr/bin/env python

import os, argparse

parser = argparse.ArgumentParser();

parser.add_argument('-f', '--folder')
parser.add_argument('-c', '--command')
parser.add_argument('-t', '--trigger', required=False, default='{{file}}');

opts = parser.parse_args();

for f in os.listdir(opts.folder):
    cmd = opts.command.replace(opts.trigger, os.path.join(opts.folder, f));
    print(cmd)
    os.system(cmd)
