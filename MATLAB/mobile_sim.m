% mobile_sim
% simulation for mobile Syk particles

gam = [0.05 0.1]; % binding rates
lam = [0.1 0.02]; % unbinding rates
D = 1; % diffusion rate slower than the background
T = 900; % number of frames
PSFsigma = 1; % psf sigma in pixels 
ROIsize = [100 100];
MeanPhotons = 200; % photons per frame
minStart = 5; % minimum number of particles to have when movie starts

% get a population of particles and their birth starts
sample = max(ceil(T*gam+5*sqrt(T*gam)));
% generate random numbers and start times for births
bran = rand(sample,2);
tact=-log(1-bran)./(ones(sample,1)*gam);
tact = cumsum(tact);
qq = sort(tact(:));
startT = qq(minStart);
tact = tact-startT;
tact(tact<1) = 1;
tact(tact>T+1) = 0;
sact = cell(2,1);
sact{1} = tact(tact(:,1)>0,1);
sact{2} = tact(tact(:,2)>0,2);
% generate random numbers and end times for births
eact = cell(2,1);
sz = sum(tact>0);
dran = {rand(sz(1),1),rand(sz(2),1)};
eact{1} = -log(1-dran{1})/lam(1);
eact{1} = eact{1}+sact{1};
eact{2} = -log(1-dran{2})/lam(2);
eact{2} = eact{2}+sact{2};
% generate starting particle positions (they will move)
spos = cell(2,1);
spos{1}=rand(sz(1),2).*(ones(sz(1),1)*ROIsize);
spos{2}=rand(sz(2),2).*(ones(sz(2),1)*ROIsize);
% convert data into a tracks cell array
coords = cell(2,1);
coords{1} = cell(sz(1),1);
coords{2} = cell(sz(2),1);
% make tracks array for two populations of particles moving with a d value
for ii = 1:sz(1)
    sframe = floor(sact{1}(ii));
    elen = ceil(eact{1}(ii))-sframe+1;
    coords{1}{ii} = zeros(elen,4);
    coords{1}{ii}(:,[1,2]) = ones(elen,1)*spos{1}(ii,:)...
        +cumsum(sqrt(2*D)*randn(elen,2));
    coords{1}{ii}(:,3) = ones(elen,1)*MeanPhotons;
    coords{1}{ii}(:,4) = (sframe:sframe+elen-1)';
    coords{1}{ii}(1,4) = sact{1}(ii);
    coords{1}{ii}(end,4) = eact{1}(ii);
end

for ii = 1:sz(2)
    sframe = floor(sact{2}(ii));
    elen = ceil(eact{2}(ii))-sframe+1;
    coords{2}{ii} = zeros(elen,4);
    coords{2}{ii}(:,[1,2]) = ones(elen,1)*spos{2}(ii,:)...
        +cumsum(sqrt(2*D)*randn(elen,2));
    coords{2}{ii}(:,3) = ones(elen,1)*MeanPhotons;
    coords{2}{ii}(:,4) = (sframe:sframe+elen-1)';
    coords{2}{ii}(1,4) = sact{2}(ii);
    coords{2}{ii}(end,4) = eact{2}(ii);
end

% create a cell array of Frame Data from two populations
% put the subsequent coordinates and times in columns 5 6 and 7
modCoords = coords;
for ii = 1:sz(1)
    modCoords{1}{ii} = [modCoords{1}{ii}(1:end-1,:) modCoords{1}{ii}(2:end,[1 2 4])];
end
for ii = 1:sz(2)
    modCoords{2}{ii} = [modCoords{2}{ii}(1:end-1,:) modCoords{2}{ii}(2:end,[1 2 4])];
end
% put all tracks into a matrix
tempMat = cell2mat(modCoords{1});
tempMat = [tempMat; cell2mat(modCoords{2})];            
% put all coordinates into the frame data format for image building
FrameDat = cell(T,1);
for ii = 1:T
    % get coordinates of frame of interest
    ind = floor(tempMat(:,4)) == ii;
    FrameDat{ii} = tempMat(ind,:);
end

% get emission times of photons for each blob
N = sum(sz);
emitTime = cell(T,N);
% loop over each frame and determine photon emission times
for ii = 1:T
    tempFrame = FrameDat{ii};
    for jj = 1:size(tempFrame,1) % loop over emitters
        % sample out 5 sigmas of random numbers from the max rate
        % speed is more important than memory management here
        Rate = MeanPhotons;
        sample = ceil(Rate+5*sqrt(Rate));
        CDFval = rand(sample,1); % get all the emission CDF times
        % convert the CDFvalues to the wait times and do a
        % cumulative sum to get emission times
        xx = -log(1-CDFval)/Rate;
        xx = cumsum(xx);
        % remove anything greater than the frame time
        xx(xx > (tempFrame(jj,7)-tempFrame(jj,4))) = [];
        emitTime{ii,jj} = xx;
    end
end

% get photon positions to put on the ROI
photPos = cell(T,N);
% loop over each frame and determine photon positions
for ii = 1:T
    tempFrame = FrameDat{ii};
    for jj = 1:size(tempFrame,1) % loop over emitters
        timevec = emitTime{ii,jj};
        photcount = length(timevec);
        if photcount == 0
            continue; % ignore non-emission states
        end
        frameInput = tempFrame(jj,:);
        % calculate the positions due to an interpolated
        % brownian bridge
        delt = frameInput(7)-frameInput(4);
        A = [frameInput(1) frameInput(2)];
        B = [frameInput(5) frameInput(6)];        
        dL = (diff([0; timevec]));        
        bvar = sqrt(2*D*dL*ones(1,2)).*randn(length(dL),2);
        intpos = cumsum(bvar)+ones(size(bvar,1),1)*A;
        tint = timevec/delt;
        maxDisp = intpos(end,:)-B;
        intpos = intpos - tint*maxDisp;
        % add diffraction based randomness
        xx = intpos(:,1)+PSFsigma*randn(photcount,1);
        yy = intpos(:,2)+PSFsigma*randn(photcount,1);
        % periodic Boundary Conditions around ROI
        xx = mod(xx,ROIsize(1));
        yy = mod(yy,ROIsize(2));
        photPos{ii,jj} = [xx yy];
    end
end

% create a movie from photon positions
% initialize the image matrix
imMat3 = zeros(ROIsize(1), ROIsize(2), T);

% loop over each frame
for ii = 1:T
    tempFrame = FrameDat{ii}; % use to parse emitter counts
    if isempty(tempFrame)
        continue;
    end
    % loop over each emitter per frame
    M = sparse(ROIsize(1),ROIsize(2),1); % clear M on every iteration
    for jj = 1:size(tempFrame,1)
        if isempty(photPos{ii,jj})
            continue;
        end
        pixelPos = zeros(size(photPos{ii,jj},1)+1,2);
        pixelPos = pixelPos(1:end-1,:) + ceil(photPos{ii,jj}); % center of first pixel is 0.5
        pixelPos(end,:) = [ROIsize(1), ROIsize(2)]; % make the matrix image sized
        % remove photons that don't hit the ROI
        remind = pixelPos(:,1) < 1 | pixelPos(:,1) > ROIsize(1);
        pixelPos(remind,:) = [];
        remind = pixelPos(:,2) < 1 | pixelPos(:,2) > ROIsize(2);
        pixelPos(remind,:) = [];
        % build up the photon matrix
        M = M + sparse(pixelPos(:,1),pixelPos(:,2),1);
    end
    % add photons to the image
    imMat3(:,:,ii) = imMat3(:,:,ii) + M;
    % remove extra counts placed in the corner
    imMat3(ROIsize(1),ROIsize(2),ii) = imMat3(ROIsize(1),ROIsize(2),ii) - jj - 1;
end
