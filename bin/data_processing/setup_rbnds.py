import sys

if __name__ == "__main__":
    exp_mat_file = sys.argv[1]
    cluster_file = sys.argv[2]
    outfile = sys.argv[3]

    point2cluster = dict()
    with open(cluster_file,'r') as fin:
        for line in fin:
            splt = line.strip().split("\t")
            point2cluster[splt[0]] = splt[1]

    first=True
    with open(outfile,'w') as fout:
        with open(exp_mat_file,'r') as fin:
            for line in fin:
                if not first:
                    splt = line.strip().split(" ")
                    mid = splt[0]
                    cid = point2cluster[mid]
                    vec = splt[1:]
                    fout.write("{}\t{}\t{}\n".format(mid,cid,"\t".join(vec)))
                else:
                    first = False






