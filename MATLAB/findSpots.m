function [coordinates, LoGs] = findSpots(im, sigmas, Thresh)
% findSpots: A function to take an arbitrary movie and find blob features
%   inputs: im: input movie (nxmxt matrix, t>5)
%           sigmas: vector of PSFsigmas
%           Thresh: threshold for found maxima (of input movie pixels)
%   output: coordinates (x,y,t,s,maxima)

%% input checks
if nargin < 2
    sigmas = [1,1.5]; % sigma sizes for scale space LoG filter
end
if nargin < 3
    % final threshold for selecting maxima centers
    Thresh = 5;
end
% calculate and output raw coordinates
[coordinates, LoGs] = findLoGMinima(im,sigmas,1);
% filter coordinates
coordinates = applyThreshold(Thresh,coordinates);
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
% Vector of Laplacian of Gaussian filters, based on range of sigmas
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
function [coordinates, LoGs] = findLoGMinima(movie,sigmas,order)
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
    % get max filter of movie to negate noise for filtering
    maxMovie = movie;
    for ii = 1:size(movie,3)
        maxMovie(:,:,ii) = ordfilt2(movie(:,:,ii),9,true(3));
    end
    for ii = 1:lz(4)
       mVal = [mVal; maxMovie(squeeze(Candidates(:,:,:,ii)))];
    end
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
function filterCoordinates = applyThreshold(Threshold,coordinates)
    boolCand = coordinates(:,5) > Threshold;
    filterCoordinates = coordinates(boolCand,:);
end

