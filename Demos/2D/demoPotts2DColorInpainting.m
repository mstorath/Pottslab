% This is a demonstration of inpainting using the Potts model.
% Note that it is an intrinsic property of the Potts model to 
% create two triple junctions instead of a four-junction in the center.

% load image
img = double(imread('colors.png'))/255;

% set inpainting mask
simg = sum(img, 3);
weights = simg ~= 0;

% add noise
rng(123) % set seed value of random number gen. for reproducibility
imgNoisy = img + 0.3 * randn(size(img)) .* cat(3, weights, weights, weights);

%% Potts restoration
gamma = 2;
clear opts;
opts.weights = weights; 
opts.verbose = true;
opts.muStep = 1.1;
opts.muInit = 1e-4;
tic
u = minL2Potts2DADMM(imgNoisy, gamma, opts);
toc

%% Show result
subplot(1,2,1)
imshow(imgNoisy)
title('Noisy data (Black pixels are missing')
subplot(1,2,2)
imshow(u)
title('Potts inpainting')