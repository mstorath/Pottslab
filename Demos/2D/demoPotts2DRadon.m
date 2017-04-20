% This is a demonstration of Radon inversion from few angles using the Potts model.
% Tested on Mac OS 10.9, Matlab 2015a, 
% If everything worked correctly a result as 
% in results/plFigPottsRec7Angles.fig should appear
% Warning: This experiment can take more than one hour

% load Shepp-Logan
img = phantom(256);

% Radon transform with 7 equidistant projection angles 
nAngles = 7; 
theta = (1:nAngles)/nAngles * 180;
A = radonop(theta, size(img,1));
data = A * img;

% raw filtered backprojection and a regularized one
uFbp = iradon(data, theta, 'Linear', 'Ram-Lak', 1, size(img, 1));
uFbpReg = iradon(data, theta, 'Linear', 'Hamming', 0.3, size(img, 1));

% joint reconstruction and segmentation using the Potts model
gamma = 0.04;
uPotts = minL2iPotts2DADMM(data, gamma, A, 'verbose', 1, 'groundTruth', img, 'isotropic', 1 );

% show results
subplot(2,3,1)
imshow(img)
title('Original')
subplot(2,3,2)
imagesc(data)
title('Data (Sinogram)')
subplot(2,3,4)
imshow(uFbp)
title('FBP result')
subplot(2,3,5)
imshow(uFbpReg)
title('FBP with Hamming window')
subplot(2,3,6)
imshow(uPotts)
title('Potts result')
