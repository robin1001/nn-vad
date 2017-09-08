#!/usr/bin/python

# Created on 2016-07-10
# Author: Zhang Binbin

import sys

def print_segment(cur, start, end, info):
    print "\t\tintervals [%d]:" % cur
    print "\t\t\txmin = %f" % start
    print "\t\t\txmax = %f" % end
    print "\t\t\ttext = \"%s\"" % info

for line in sys.stdin:
    arr = line.strip().split()
    if len(arr) < 1: break
    assert((len(arr) - 1) % 4 == 0)
    end_time = int(arr[-2])
    intervals = ((len(arr) - 1) / 4) * 2 - 1;

    print "File type = \"ooTextFile\""
    print "Object class = \"TextGrid\""
    print "xmin = 0"
    print "xmax = %f" % (end_time / 100.0)
    print "tiers? <exists>"
    print "size = 1"
    print "item []:"
    print "\t item [1]:"
    print "\t\tclass = \"IntervalTier\""
    print "\t\tname = \"%s\"" % arr[0]
    print "\t\txmin = 0" 
    print "\t\txmax = %f" % (end_time / 100.0)
    print "\t\tintervals: size = %d " % intervals
    cur = 1
    if len(arr) >= 5 and int(arr[2]) != 0:
        print_segment(cur, 0, int(arr[2]) / 100.0, "N")
        cur += 1
    
    for i in range(1, len(arr), 4):
        print_segment(cur, int(arr[i+1]) / 100.0, int(arr[i+2]) / 100.0, "V")
        cur += 1
        if i < len(arr) - 4:
            print_segment(cur, int(arr[i+2]) / 100.0, int(arr[i+5]) / 100.0, "N")
            cur += 1

