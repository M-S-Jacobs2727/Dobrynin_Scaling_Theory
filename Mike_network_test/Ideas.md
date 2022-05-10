# General thoughts on procedure, model, etc.

1. There are two methods we thought of using. The first is to break the whole
($\varphi$, $N_w$, $\eta_{sp}$) space into discrete voxels, whose values are
either 0 (the surface does not go through the voxel) or 1 (it does). The 
second is to break only the domain ($\varphi$, $N_w$) space into pixels, whose 
values are the normalized $\eta_{sp}$ values. 
2. After implementing the latter, we've noticed that our mean errors are 
massive, >100% for the third parameter. This has led me to believe that the 
former is the better option, as it better represents the classic ML case of 
finding an edge in an image.
3. In determining the value of the voxels in the first case, we can simply
compute the values of $\eta_{sp}$ at the corner with the lower ($\varphi$, 
$N_w$) values and with the higher ($\varphi$, $N_w$) values, since $\eta_{sp}$ 
scales monotonically with both.
4. By generating the surfaces directly on the GPU with torch, I can save time
in transferring the data across the bus, in addition to the raw speedup that
the GPU provides. With 32 CPU cores, I was generating, transferring, and 
learning the data at ~1M surfaces every 5 minutes at a resolution of (32, 32). 
With torch, I can do the same at around ~1M surfaces every 3-5 seconds.
5. For the voxel method, we'll have to be careful about defining the edge
of the voxel vs. the center of the voxel.