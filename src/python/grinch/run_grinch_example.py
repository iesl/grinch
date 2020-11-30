

from grinch.grinch_alg import Grinch
import numpy as np
import wandb

if __name__ == "__main__":
    wandb.init()
    point_labels = np.random.random_integers(0,10,100)
    vectors = np.random.random((100, 5)).astype(np.float32)
    grinch = Grinch(points=vectors)
    grinch.build_dendrogram()
    grinch.write_tree('tmp.tree.out', point_labels)