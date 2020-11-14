function [coordinates,filterMovies] = cega( im, ConnThresh, WinLength, sigmas, KLThresh )
%cega: A function to take a motor movie and find the moving components
%  and output them as an augmented movie
% inputs: im : input movie (nxmxt matrix, t>5 for this to work)
%         ConnThresh: threshold of KL divergence values for connectivity filter
%         WinLength: sliding window length for median filter
% output: filterMovies: structure of matrices of various filtered movies
%   Author: PKR UPENN JUNE 2020
% ce:ga (Tohono O'odham: To find object)
% License: GPL-3.0-or-later

%% input checks
if nargin < 2
    % threshold for connectivity filter
    ConnThresh = 0.5;
end
if nargin < 3
    WinLength = 31; % 31 frame temporal median filter!
end
if nargin < 4
    sigmas = [1,1.5]; % sigma sizes for scale space LoG filter
end
if nargin < 5
    KLThresh = 0.75;
end
% get filtered movies
filterMovies = getFilterMovies(im,ConnThresh,WinLength);
% calculate and output raw coordinates
%coordinates = findLoGMinima(filterMovies.LoG,sigmas,1);
%coordinates = findLoGMinima(filterMovies.expectedP,sigmas,1);
coordinates = findLoGMinima(filterMovies.ConnectivityFilter,sigmas,1);
% filter coordinates
coordinates = applyThreshold(KLThresh,coordinates);
end

% high level function for filter movies
function filterMovies = getFilterMovies(im,ConnThresh,WinLength)
% get sizes
imsize = size(im);
if length(imsize) > 2 % no single frame inputs
    frameT = imsize(3);
else
    error('This method only works on an image stack (3D-Matrix)');
end
%% no negative pixels
imM = single(im);
imM(imM<=0) = 0.01;
%% generate Gaussian filtered movies with sigma 1 and 2
gKernel1 = gaussKernel(1,7);
gKernel2 = gaussKernel(2,11);
tKernel = gaussKernel(1,5);
[d2gKernel,~] = d2GaussKernel(1,7);
Gauss1 = filter2D(imM,gKernel1,gKernel1);
Gauss2 = filter2D(imM,gKernel2,gKernel2);
%% <Q>: prior movie with a temporal median filter
iBlur = tKernel(1)*Gauss2+tKernel(2)*Gauss1+tKernel(3)*imM ...
    +tKernel(4)*Gauss1+tKernel(5)*Gauss2;
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
MotionMovie = (tKernel(3)*imM+tKernel(4)*Gauss1(:,:,pt)+tKernel(2)*Gauss1(:,:,mt)+...
        tKernel(1)*Gauss2(:,:,mmt)+tKernel(5)*Gauss2(:,:,ppt));
%% kl-divergence movie of <Q> from <P>
% add 0.001 to deal with rounding errors for KL values near 0
KLMovie = MotionMovie.*(-log(PriorMovie)+log(MotionMovie)) - MotionMovie + PriorMovie + 0.001;
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
% Generate a LoG movie
LoG = LoGFilter(ConnFilt, gKernel1, d2gKernel);
% output movies
filterMovies = struct('expectedQ',PriorMovie,'expectedP',MotionMovie,...
    'KLM',KLMovie,'ConnectivityFilter',ConnFilt,'LoG',LoG);
end

% generates a 1D normalized gaussian kernel, always use an odd sz
function kernel = gaussKernel(sigma,sz) % sampled Gaussian
    pxrange = (1:sz) - ceil(sz/2);
    kernel = exp(-(pxrange.^2)/2/sigma/sigma);
    kernel = kernel/sum(kernel);
end
function kernel = errKernel(sigma,sz) % integrated Gaussian
    pxrange = (1:sz) - ceil(sz/2);
    K0 = erf((pxrange-0.5)/sigma/sqrt(2));
    K1 = erf((pxrange+0.5)/sigma/sqrt(2));
    kernel = (K1-K0)/sum(K1-K0);
end
function kernel = disGKernel(sigma,sz) % discrete Diffusion
    pxrange = (1:sz) - ceil(sz/2);
    t = sigma*sigma;
    kernel = exp(-t)*besseli(pxrange,t);
    kernel = kernel/sum(kernel);
end
% applies a 1D kernel in the x and y axis of a movie, sequentially
function filterOut = filter2D(movie,kernelX,kernelY)
    kX = length(kernelX);
    midX = ceil(kX/2);
    kY = length(kernelY);
    midY = ceil(kY/2);
    filter1 = 0*movie;
    filterOut = 0*movie;
    mz = size(movie);
    % convolve vertically first
    for jj = 1:kX
        lb = max(midX-jj+1,1);
        rb = min(mz(1),mz(1)-jj+midX);
        ml = max(jj-midX+1,1);
        mr = min(mz(1),mz(1)-midX+jj);
        filter1(lb:rb,:,:) = filter1(lb:rb,:,:) + ...
        movie(ml:mr,:,:)*kernelX(jj);
    end
   % reflect edges fix this expression
    for jj = 1:midX-1
        for kk = 1:midX-jj % loop over missed kernel segments
        filter1(jj,:,:) = filter1(jj,:,:) + ...
            filter1(kk,:,:)*kernelX(midX-kk-jj+1);
        filter1(end-jj+1,:,:) = filter1(end-jj+1,:,:) + ...
            filter1(end-kk+1,:,:)*kernelX(midX+kk+jj-1);
        end
    end
   % convolve horizontally next
    for jj = 1:kY
        lb = max(midY-jj+1,1);
        rb = min(mz(2),mz(2)-jj+midY);
        ml = max(jj-midY+1,1);
        mr = min(mz(2),mz(2)-midY+jj);
        filterOut(:,lb:rb,:) = filterOut(:,lb:rb,:) + ...
        filter1(:,ml:mr,:)*kernelY(jj);
    end
    % reflect edges
    for jj = 1:midY-1
        for kk = 1:midY-jj % loop over missed kernel segments
        filterOut(:,jj,:) = filterOut(:,jj,:) + ...
            filterOut(:,kk,:,:)*kernelY(midY-kk-jj+1);
        filterOut(:,end-jj+1,:) = filterOut(:,end-jj+1,:) + ...
            filterOut(:,end-kk+1,:)*kernelY(midY+kk+jj-1);
        end
    end
end
% 2nd derivative of sampled Gaussian Kernel, multiplied by sig^2 for scale invariance
function [kernel, gKernel] = d2GaussKernel(sigma,sz)
    gKernel = gaussKernel(sigma,sz);
    pxrange = (1:sz) - ceil(sz/2);
    kernel = gKernel.*((pxrange.^2-sigma^2)/sigma^2);
end
% Laplacian of the Gaussian Filter
function LoG = LoGFilter(movie, gKernel, d2gKernel)
    dxx = filter2D(movie, d2gKernel, gKernel);
    dyy = filter2D(movie, gKernel, d2gKernel);
    LoG =  dxx+dyy;
end
function LoGstack = scaleSpaceLoG(movie,sigmas)
    mz = size(movie);
    numSigs = length(sigmas);
    LoGstack = zeros([mz numSigs]);
    for ii = 1:numSigs
        kz = 6*sigmas(ii)+1;
        kz = kz + mod(kz+1,2); % must always have odd length kernel
        [d2gKernel,gKernel] = d2GaussKernel(sigmas(ii),kz);
        LoGstack(:,:,:,ii) = LoGFilter(movie,gKernel,d2gKernel);
    end
end
% get raw maxima coordinates from LoG movie here
function coordinates = findLoGMinima(movie,sigmas,order)
    if nargin < 3
        order = 1;
    end
    LoGs = scaleSpaceLoG(movie, sigmas);
    Candidates = LoGs<0;
    % order is nearest neighbor distance
    for jj = 1:order
        % x-axis
        Candidates(1:end-jj,:,:,:) = Candidates(1:end-jj,:,:,:) ...
        & LoGs(1:end-jj,:,:,:) < LoGs(1+jj:end,:,:,:);
        Candidates(1+jj:end,:,:,:) = Candidates(1+jj:end,:,:,:) ...
        & LoGs(1+jj:end,:,:,:) < LoGs(1:end-jj,:,:,:);
        % y-axis
        Candidates(:,1:end-jj,:,:) = Candidates(:,1:end-jj,:,:) ...
        & LoGs(:,1:end-jj,:,:) < LoGs(:,1+jj:end,:,:);
        Candidates(:,1+jj:end,:,:) = Candidates(:,1+jj:end,:,:) ...
        & LoGs(:,1+jj:end,:,:) < LoGs(:,1:end-jj,:,:);
        % scale-axis
        Candidates(:,:,:,1:end-jj) = Candidates(:,:,:,1:end-jj) ...
        & LoGs(:,:,:,1:end-jj) < LoGs(:,:,:,1+jj:end);
        Candidates(:,:,:,1+jj:end) = Candidates(:,:,:,1+jj:end) ...
        & LoGs(:,:,:,1+jj:end) < LoGs(:,:,:,1:end-jj);
    end
    % loop over remaining elements and check all "surviving" nearest neighbors
    lz = size(Candidates);
    CandInd = find(Candidates);
    [x,y,t,s] = ind2sub(lz,CandInd);
    mVal = [];
    % get sum of movie adjacent neighbors to augment second threshold of
    % connectivity filter movie
    sumMovie = movie;
    sumMovie(1:end-1,:,:) = sumMovie(1:end-1,:,:) + movie(2:end,:,:);
    sumMovie(2:end,:,:) = sumMovie(2:end,:,:) + movie(1:end-1,:,:);
    sumMovie(:,1:end-1,:) = sumMovie(:,1:end-1,:) + movie(:,2:end,:);
    sumMovie(:,2:end,:,:) = sumMovie(:,2:end,:) + movie(:,1:end-1,:);
    sumMovie(1:end-1,1:end-1,:) = sumMovie(1:end-1,1:end-1,:) + movie(2:end,2:end,:);
    sumMovie(2:end,2:end,:) = sumMovie(2:end,2:end,:) + movie(1:end-1,1:end-1,:);
    sumMovie(2:end,1:end-1,:) = sumMovie(2:end,1:end-1,:) + movie(1:end-1,2:end,:);
    sumMovie(1:end-1,2:end,:,:) = sumMovie(1:end-1,2:end,:) + movie(2:end,1:end-1,:);
    for ii = 1:lz(4)
       mVal = [mVal; sumMovie(squeeze(Candidates(:,:,:,ii)))];
    end
%     for ii = 1:lz(4)
%        mVal = [mVal; movie(squeeze(Candidates(:,:,:,ii)))];
%     end
    BoolCand = true(length(CandInd),1);
    for ii = 1:length(CandInd)
        val = LoGs(x(ii),y(ii),t(ii),s(ii));
        s_l = max(1,s(ii)-order);
        s_r = min(lz(4),s(ii)+order);
        y_l = max(1,y(ii)-order);
        y_r = min(lz(2),y(ii)+order);
        x_l = max(1,x(ii)-order);
        x_r = min(lz(1),x(ii)+order);
        batch = LoGs(x_l:x_r,y_l:y_r,t(ii),s_l:s_r)-val;
        if sum(batch(:)<=0) > 1
            BoolCand(ii) = false;
        end
    end 
    coordinates = [x(BoolCand),y(BoolCand),t(BoolCand),s(BoolCand),mVal(BoolCand)];
end
% thresholding function
function filterCoordinates = applyThreshold(KLThreshold,coordinates)
    boolCand = coordinates(:,5) > KLThreshold;
    filterCoordinates = coordinates(boolCand,:);
end
