% cegaSim.m
% script to simulate motors on an experimental background
% Author: PKR, July 2020

%% Random number seed, change this to change the simulation random numbers
% comment out this variable to make the simulation different every time
rng(8675309);

%% Set simulation parameters here
% experimental parameters for the background
gain = 70; % effective photon gain (ADU/e-)
offset = 2293; % ADU value added to the image
WinLength = 31; % tempoeral window for median filter, must be an odd number
% Simulation Parameters - vector number of populations
N = [5,5]; % Number of motors for each population per frame
D = [0.5,0.5]; % diffusion rate by population, isotropic diffusion
V = [0.63,0.59]; % mean speed of each motor population
VMax = [2.0,2.0]; % maximum vector component speed of motor populations (confied process)
InitSigV = [0.5,0.5]; % lognormal sigma of the change in population speeds
TimeSigV = [0.01,0.01]; % sigma of the change in population speeds, gaussian process
% Photon Painting Parameters, Currently Uniform Across Populations
PSFsigma = 0.75; % psf sigma in pixels, undersampled to match data
MeanPhotons = 150; % expected photons per motor per frame
DBridge = 0.5; % diffusion constant of the brownian bridge between observations
WhiteNoiseSigma = 0.1; % sigma for white noise to make everything more realistic
%% get pop. numbers from N vector length, all sim params should have same vector length
numPops = length(N);

% prompt for the file location
[fileName,pathLoc] = uigetfile('.tif','Select a tif stack to use as the BG/ROI');
fileLoc = fullfile(pathLoc,fileName);

% Load the tif file
InfoImage=imfinfo(fileLoc);
mImage=InfoImage(1).Width;
nImage=InfoImage(1).Height;
NumberImages=length(InfoImage);
Vector=zeros(nImage,mImage,NumberImages,'uint16');
for i=1:NumberImages
   Vector(:,:,i)=imread(fileLoc,'Index',i);
end

% Gain and Offset Adjustment
im = (single(Vector)-offset)/gain;
im(im<=0) = 0.01; % no negative numbers
sz = size(im);
% create background movie with temporal filter
HalfWin = floor(WinLength/2);
bg = 0*im;
for ii = 1:HalfWin
    bg(:,:,ii) = median(im(:,:,1:ii+HalfWin),3);
end
for ii = HalfWin+1:sz(3)-HalfWin-1
    bg(:,:,ii) = median(im(:,:,ii-HalfWin:ii+HalfWin),3);
end
for ii = sz(3)-HalfWin:sz(3)
    bg(:,:,ii) = median(im(:,:,ii-HalfWin:end),3);
end

%% generate a set of tracks for the moving motors
% when a motor leaves the ROI, kill the track and start a new one
numFrames = sz(3);
yMax = sz(2); % get ROI boundaries, setting x axis = 1, y axis = 2
xMax = sz(1);
%% pick starting positions and velocities of motors
NStarts = cell(numPops,1);
vNStarts = cell(numPops,1);
for ii = 1:numPops
    NStarts{ii} = rand(N(ii),2).*repmat([xMax,yMax],N(ii),1);
    vNStarts{ii} = lognrnd(log(V(ii)),InitSigV(ii),N(ii),2);
    vNStarts{ii}(NStarts{ii}(:,1) > xMax/2,1) = ...
        -vNStarts{ii}(NStarts{ii}(:,1) > xMax/2,1);
    vNStarts{ii}(NStarts{ii}(:,2) > yMax/2,2) = ...
        -vNStarts{ii}(NStarts{ii}(:,2) > yMax/2,2);
end
%% track trajectories, measured before and after each frame
NTracks = cell(numPops,1);
Nactive = cell(numPops,1);
% Start initial track structures
% Tracks have variable length so use cell array of matrices
for ii = 1:numPops
    tempTracks = {};
    for jj = 1:N(ii)
        tempTracks{jj} = [NStarts{ii}(jj,:),0,vNStarts{ii}(jj,:)];
    end
    NTracks{ii} = tempTracks;
    Nactive{ii}{1} = 1:N(ii);
end

% loop over entire movie length and generate tracks
nN = N+1; % next track element for each population

for ii = 1:numPops
    for tt = 1:sz(3)
        Nactive{ii}{tt+1} = [];
        for elem = Nactive{ii}{tt}
            nCoordinates = NTracks{ii}{elem}(end,:);
            % drift w/ isotropic diffusion for position
            nCoordinates(1) = nCoordinates(1) + nCoordinates(4) + ...
                sqrt(2*D(ii))*randn();
            nCoordinates(2) = nCoordinates(2) + nCoordinates(5) + ...
                sqrt(2*D(ii))*randn();
            nCoordinates(3) = tt;
            % isotropic confined diffusion for velocity
            nCoordinates(4) = nCoordinates(4)+TimeSigV(ii)*randn();
            nCoordinates(4) = max(-VMax(ii),min(VMax(ii),nCoordinates(4)));
            nCoordinates(5) = nCoordinates(5)+TimeSigV(ii)*randn();
            nCoordinates(5) = max(-VMax(ii),min(VMax(ii),nCoordinates(5)));
            NTracks{ii}{elem} = [NTracks{ii}{elem}; nCoordinates];
            % kill a track if it wanders out of ROI, start a new track
            if (nCoordinates(1) < 0 || nCoordinates(1) > sz(1) || ...
                nCoordinates(2) < 0 || nCoordinates(2) > sz(2))
                Nactive{ii}{tt+1} = [Nactive{ii}{tt+1}, nN(ii)];
                newNStart = rand(1,2).*[xMax,yMax];
                newNVelocity = lognrnd(log(V(ii)),InitSigV(ii),1,2);
                newNVelocity(newNStart(1) > xMax/2,1) = ...
                    -newNVelocity(newNStart(1) > xMax/2,1);
                newNVelocity(newNStart(2) > yMax/2,2) = ...
                    -newNVelocity(newNStart(2) > yMax/2,2);
                NTracks{ii}{nN(ii)} = [newNStart, tt, newNVelocity];
                nN(ii) = nN(ii)+1;
            else
                Nactive{ii}{tt+1} = [Nactive{ii}{tt+1}, elem];
            end
        end
    end
end
%% get emission times of photons for each blob
emissionTimes = cell(numPops,1);
for ii = 1:numPops
    emissionPop = {};
    for mm = 1:nN(ii)-1
        startTime = NTracks{ii}{mm}(1,3);
        endTime = NTracks{ii}{mm}(end,3);
        TimeGap = endTime-startTime;
        sampleSize = ceil(MeanPhotons*TimeGap + 5*sqrt(MeanPhotons*TimeGap));
        timeCDFs = rand(sampleSize,1);
        % convert CDFs to emission times
        xx = -log(1-timeCDFs)/MeanPhotons;
        xx = cumsum(xx)+startTime;
        xx = xx(xx<endTime);
        emissionPop{mm} = xx;
    end
    emissionTimes{ii} = emissionPop;
end

%% paint photons from tracks and emission time vectors
%% output a tracks array, localization list, and photon movie
motorPhotons = zeros(sz);
SimTracks = cell(numPops,1);
SimLocalizations = [];

for ii = 1:numPops
    SimTracks{ii} = {};
    for mm = 1:nN(ii)-1 % last nN is never used
        SimTracks{ii}{mm} = [];
        cc = 1; % counter for emission times
        trackStartTime = floor(NTracks{ii}{mm}(1,3));
        numEmissions = length(emissionTimes{ii}{mm});
        PhotonDeviations = PSFsigma*randn(numEmissions,2);
        BBridgeDraws = randn(numEmissions,2);
        while cc < numEmissions
            startEmission = cc;
            tt = ceil(emissionTimes{ii}{mm}(cc));
            flag = true;
            while flag
                cc=cc+1;
                try
                    flag = ceil(emissionTimes{ii}{mm}(cc))==tt;
                catch ME
                    flag = false;
                end
            end
            startPos = NTracks{ii}{mm}(tt-trackStartTime,1:2);
            endPos =  NTracks{ii}{mm}(tt-trackStartTime+1,1:2);
            % get time gaps between emission events 
            emitFrame = emissionTimes{ii}{mm}(startEmission:cc-1)-tt+1;
            dtau = diff([0; emitFrame]);
            endTau = 1 - emitFrame(end);
            % create brownian bridge deviations
            BBridgeDeviations = repmat(sqrt(dtau*2*DBridge),1,2).* ...
                BBridgeDraws(startEmission:cc-1,:);
            particlePosition = cumsum(BBridgeDeviations)+startPos;
            % sample a jump to the end of the frame
            maxDisplacement = sqrt(endTau*2*DBridge)*randn(1,2)+ ...
                particlePosition(end,:)-endPos;
            % brownian bridge constraint on positions
            particlePosition = particlePosition - ... 
                [emitFrame*maxDisplacement(1),emitFrame*maxDisplacement(2)];
            % draw photon positions from PSFsigma deviation
            photonPosition = particlePosition + ...
                PhotonDeviations(startEmission:cc-1,:);
            % throw out photons outside of the ROI
            photonPosition = photonPosition(photonPosition(:,1)>0,:);
            photonPosition = photonPosition(photonPosition(:,2)>0,:);
            photonPosition = photonPosition(photonPosition(:,1)<sz(1),:);
            photonPosition = photonPosition(photonPosition(:,2)<sz(2),:);
            frameCoordinates = [mean(photonPosition,1),tt,length(photonPosition)];
            % put photon averages into simulated tracks and localization structures
            SimLocalizations = [SimLocalizations; frameCoordinates];
            SimTracks{ii}{mm} = [SimTracks{ii}{mm}; frameCoordinates];
            % paint the motorPhotons movie
            pixelPosition = ceil(photonPosition);
            for jj = 1:size(photonPosition,1)
                motorPhotons(pixelPosition(jj,1),pixelPosition(jj,2),tt) = ...
                motorPhotons(pixelPosition(jj,1),pixelPosition(jj,2),tt) + 1;
            end
        end
    end
end

% create poisson-like realization of the background
bgRealization = poissrnd(bg);
finalMovie = WhiteNoiseSigma*randn(sz)+bgRealization+motorPhotons;
