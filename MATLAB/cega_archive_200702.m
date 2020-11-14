function filterMovies = cega( im, ConnThresh, WinLength )
%cega: A function to take a motor movie and find the moving components
%  and output them as an augmented movie
% inputs: im : input movie (nxmxt matrix, t>5 for this to work)
%         ConnThresh: threshold of KL divergence values for connectivity filter
%         WinLength: sliding window length for median filter
% output: filterMovies: structure of matrices of various filtered movies
%   Author: PKR UPENN JUNE 2020
% ce:ga (Tohono O'odham: To find object)

%% input checks
if nargin < 2
    % threshold for connectivity filter
    ConnThresh = 1;
end
if nargin < 3
    WinLength = 31; % 31 frame temporal median filter!
end
% get sizes
imsize = size(im);
framesize = imsize(1:2);
if length(imsize) > 2 % no single frame inputs
    frameT = imsize(3);
else
    error('This method only works on an image stack (3D-Matrix)');
end
%% no negative pixels
imM = im;
imM(imM<=0) = 0.01;
%% generate Gaussian filtered movies with sigma 1 and 2
zig1 = disGKernel(1,7);
zig2 = disGKernel(2,11);
zigt = disGKernel(1,5);
Gauss1 = filter2D(imM,zig1);
Gauss2 = filter2D(imM,zig2);
%% <Q>: prior movie with a temporal median filter
iBlur = zigt(1)*Gauss2+zigt(2)*Gauss1+zigt(3)*imM ...
    +zigt(4)*Gauss1+zigt(5)*Gauss2;
PriorMovie = imM*0;
WL = floor(WinLength/2);
WR = ceil(WinLength/2);
for ii = 1:WL
    PriorMovie(:,:,ii) = median(iBlur(:,:,1:ii+WL),3);
end
for ii = WR:size(imM,3)-WR
   PriorMovie(:,:,ii) = median(iBlur(:,:,ii-WL:ii+WL),3); 
end
for ii = size(imM,3)-WL:size(imM,3)
    PriorMovie(:,:,ii) = median(iBlur(:,:,ii-WL:end),3);
end
%% <P>: model movie with a ballistic diffusion filter
ppt = [3:frameT frameT frameT-1];
pt = [2:frameT frameT];
mt = [1 1:frameT-1];
mmt = [2 1 1:frameT-2];
MotionMovie = (zigt(3)*imM+zigt(4)*Gauss1(:,:,pt)+zigt(2)*Gauss1(:,:,mt)+...
        zigt(1)*Gauss2(:,:,mmt)+zigt(5)*Gauss2(:,:,ppt));
%% kl-divergence movie of <Q> from <P>
KLMovie = MotionMovie.*(-log(PriorMovie)+log(MotionMovie)) - MotionMovie + PriorMovie;
%% Connectivity filtering step
% Each spot value will be set to 0 if it is not adjacent to another spot
% above specific threshold
ConnFilt = KLMovie;
ThreshMovie = ConnFilt > ConnThresh;
for j = 1:(size(ConnFilt, 1))
        % set left right bounds to deal with image border
        lb = max(1,j-1);
        rb = min(size(ConnFilt,1),j+1);
    for k = 1:(size(ConnFilt, 2))
        % set top bottom bounds to deal with border
        tb = max(1,k-1);
        bb = min(size(ConnFilt,2),k+1);
        % filtering done here
        tempValids = squeeze(ThreshMovie(j,k,:));
        ThreshChunk = ThreshMovie(lb:rb,tb:bb,tempValids);
        conn = sum(sum(ThreshChunk,1),2);
        conn = conn(:);
        tempValids(tempValids) = conn > 3;
        ConnFilt(j,k,~tempValids) = 0;
    end
end
% output movies
filterMovies = struct('expectedQ',PriorMovie,'expectedP',MotionMovie,...
    'KLM',KLMovie,'ConnectivityFilter',ConnFilt);
end
% generates a 1D normalized gaussian kernel, always use an odd sz
function kernel = gaussKernel(sigma,sz)
    pxrange = 1:sz - ceil(sz/2);
    kernel = exp(-(pxrange.*2)/2/sigma/sigma);
    kernel = kernel/sum(kernel);
end
function kernel = errKernel(sigma,sz)
    pxrange = 1:sz - ceil(sz/2);
    K0 = erf((pxrange-0.5)/sigma/sqrt(2));
    K1 = erf((pxrange+0.5)/sigma/sqrt(2));
    kernel = (K1-K0)/sum(K1-K0);
end
function kernel = disGKernel(sigma,sz)
    pxrange = 1:sz - ceil(sz/2);
    t = sigma*sigma;
    kernel = exp(-t)*besseli(pxrange,t);
    kernel = kernel/sum(kernel);
end
% applies a 1D kernel in the x and y axis of a movie, sequentially
function filterOut = filter2D(mov,kernel)
    kz = length(kernel);
    mid = ceil(kz/2);
    filter1 = 0*mov;
    filterOut = 0*mov;
    mz = size(mov);
    % convolve vertically first
    for jj = 1:kz
        lb = max(mid-jj+1,1);
        rb = min(mz(1),mz(1)-jj+mid);
        ml = max(jj-mid+1,1);
        mr = min(mz(1),mz(1)-mid+jj);
        filter1(lb:rb,:,:) = filter1(lb:rb,:,:) + ...
        mov(ml:mr,:,:)*kernel(jj);
    end
   % renormalize edges
    for jj = 1:mid-1
        norml = sum(kernel(mid-jj+1:end));
        normr = sum(kernel(1:end-mid+jj));
        filter1(jj,:,:) = filter1(jj,:,:)/norml;
        filter1(end-jj+1,:,:) = filter1(end-jj+1,:,:)/normr;
    end
   % convolve horizontally next
    for jj = 1:kz
        lb = max(mid-jj+1,1);
        rb = min(mz(2),mz(2)-jj+mid);
        ml = max(jj-mid+1,1);
        mr = min(mz(2),mz(2)-mid+jj);
        filterOut(:,lb:rb,:) = filterOut(:,lb:rb,:) + ...
        filter1(:,ml:mr,:)*kernel(jj);
    end
    % renormalize edges
    for jj = 1:mid-1
        norml = sum(kernel(mid-jj+1:end));
        normr = sum(kernel(1:end-mid+jj));
        filterOut(:,jj,:) = filterOut(:,jj,:)/norml;
        filterOut(:,end-jj+1,:) = filterOut(:,end-jj+1,:)/normr;
    end
end
