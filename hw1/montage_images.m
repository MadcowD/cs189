function img=montage_images(images)
num_images=min(1000,size(images,3));
numrows=floor(sqrt(num_images));
numcols=ceil(num_images/numrows);
img=uint8(zeros(numrows*28,numcols*28));
for k=1:num_images
    [r,c]=ind2sub([numrows numcols],k);
    img((r-1)*28+1:r*28,(c-1)*28+1:c*28)=images(:,:,k);
end
