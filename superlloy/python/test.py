import sys
print "scrip:", sys.argv[0]
for i in range(1, len(sys.argv)):
    print "parameter", i, sys.argv[i]