Confocal microscopy image stacks are used to indirectly measure pectin concentration in the beads via the use of fluorescent tracers.
The method consists in singling out and labelling every tracer which allows to count them, defining $N_{tracers}$, and to compute a outline of the bead thus measuring its volume $V_{bead}$. This permits to define tracer concentration $C_{tracers}$, see equation (A). Under the assumption that the tracers and the pectin chains stay homogenously mixed, the ratio between pectin concentration and tracer concentration will remain constant throughout the gelation process.

$C_{tracers}=\frac{N_{tracers}}{V_{bead}}$ (A)

Image stacks of the beads were taken with a 20x objective, using a vertical step of either 2 or 2.4 µm.
All folowing analysis steps are performed on the whole stack at once.
For instance, the first step of the analysis process is thresholding the image using Otsu's method to determine the threshold, so in our case Otsu's method is applied to the intensity histogram of the whole image stack rather than just image by image.
The contiguous regions in the BW image stack are then labeled which allows to count them.
Again, the contiguity is to be thought of in a 3 dimensionnal way: the pixels are considered contiguous in the same image, but also over the vertical axis for neighbouring (directly on top of one another) images.
That count is a first estimate for $N_{tracers}$, that we will call "in number".
Looking at the BW image stacks reveals that some groups of closely packed tracers are labeled as unique regions when for that estimate to be completely accurate, they should be labeled as separate regions.
This is due to the thresholding step.
A second way to estimate $N_{tracers}$ is to measure the total volume of tracers in the bead $V_{tracers}$ and divide it by the volume of one tracer $V_{1~tracer}$, see equation (B).
The total volume of tracers $V_{tracers}$ being the sum of the volumes of each individual tracer, see equation (C).

$N_{tracers}=\frac{V_{tracers}}{V_{1~tracer}}$ (B)

$V_{tracers}=\sum_{i\in tracers} V_i$ (C)

Then, $V_{1~tracer}$ is estimated to be the median volume of tracers, see equation (D), under the asumption that more than half of the detected regions account for only one tracer.
This assumption can be checked visually on the binary image stack.
Plotting the size distribution of the tracer particles is also a good indicator for this asumption as the median value is read on the cumulated histogram and it should fall around the size of the tracers (1 µm).

$V_{1~tracer}= Me(V_i)_{i\in tracers}$ (D)

The volume of the bead $V_{bead}$ is estimated by computing the convex hull of the centroids of the labeled regions and computing the volume of that convex hull.
Only the centroids are used to compute the convex hull instead of the full binary image as it would increase the computation time significantly.
