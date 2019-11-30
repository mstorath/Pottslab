function labelImg = segToLabel(u, conn)
%SEGTOLABEL Converts the output image of a segmentation function (e.g. minL2Potts2DADMM) to a
%label image with integer labels (each connected component gets a unique integer)
% u: image to label (typically the output of minL2Potts2DADMM))
% conn: connectivity as in bwconncomp (default: 8)

if ~exist('conn', 'var')
    conn = 8;
end

[m,n,c] = size(u);
vals = unique(reshape(u, m*n, c), 'rows');
labelImg = zeros(m, n);
labelCounter = 0;
for i=1:size(vals,1)
    bw_mult = (vals(i,:) == reshape(u, m*n, c));
    bw = bw_mult(:,1);
    CC = bwconncomp(reshape(double(bw), m, n), conn);
    for j = 1:CC.NumObjects
        labelImg(CC.PixelIdxList{j}) = labelCounter;
        labelCounter = labelCounter + 1;
    end
end

end

