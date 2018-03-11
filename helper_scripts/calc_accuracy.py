import sys

if(len(sys.argv)!=3):
    print("Usage: calc_accuracy.py inp_file1 inp_file2")
    sys.exit(0)

total = 0
correct = 0
for l,m in zip(open(sys.argv[1],'r'),open(sys.argv[2],'r')):
    correct+=(l==m)
    total+=1
print("Accuracy: %f%%" %(1.0*correct/total*100))