"""Language-conditioned ACT for the SO-101 dataset.

Kept separate from `so101_sim` (the data-generation pipeline) on purpose: this
package only ever *reads* the HDF5 produced there, so the two can be developed
and tested independently.
"""

__version__ = "0.1.0"
