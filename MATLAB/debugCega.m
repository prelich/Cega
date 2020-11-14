% debugCega.m
% script to compare Cega filter with our tracker filter on sample data
% Author: PKR, August 2020
% License: GPL-3.0-or-Later

%% set randon number generator seed
rng(8675309);

%% Set Gain and Offset values, assuming read noise is negligible for exposition
Gain = 70;
Offset = 2293;
%% Set vector of mean photon counts for simulation studies
MeanPhotons = 150; %[50 100 150 200 250 300 350 400 450 500 550 600];
%% Call motor sim class
T = motorSim;
% set constant sim parameters
T.gain = Gain;
T.offset = Offset;
T.N = [15,15];

sigmas = [1 1.5];
WinLength = 31;

T.MeanPhotons = MeanPhotons;
% generate simulation
T.buildDataSet;
TrueCoordinates = T.SimLocalizations;

% cega inputs: movie, ConnThresh, WindowLength, sigmas, KLThreshold
[CegaCoordinates,CegaMovies] = cega(T.SimMovie,0.01,WinLength,sigmas,0.2);
% spot inputs: movie, sigmas, pixel threshold
% determine average intensity at center of psf with erf function
avIntensity = T.MeanPhotons*((erf((0.5)/sigmas(1)/sqrt(2))...
    -erf((-0.5)/sigmas(1)/sqrt(2)))/2)^2;
% threshold from 34% quantile of normal distribution at avIntensity
% add a constant to compensate for background
bgConstant = 20;
pxThresh = avIntensity + sqrt(avIntensity*2)*erfinv(2*0.34-1);
[SpotCoordinates, LoGs] = findSpots(T.SimMovie,sigmas,pxThresh+bgConstant);
% try the spot finder with background subtraction
    % estimate background movie with temporal filter
    HalfWin = floor(WinLength/2);
    Estbg = 0*T.SimMovie;
    for ii = 1:HalfWin
        Estbg(:,:,ii) = median(T.SimMovie(:,:,1:ii+HalfWin),3);
    end
    for ii = HalfWin+1:obj.sz(3)-HalfWin-1
        Estbg(:,:,ii) = median(T.SimMovie(:,:,ii-HalfWin:ii+HalfWin),3);
    end
    for ii = obj.sz(3)-HalfWin:obj.sz(3)
        Estbg(:,:,ii) = median(T.SimMovie(:,:,ii-HalfWin:end),3);
    end
[BgSpCoordinates, ~] = findSpots(T.SimMovie-Estbg,sigmas,pxThresh);

%% Paint coordinates onto the final image to see how identification looks
% cega coordinates
xC = cegaCoordinates(:,1);
yC = cegaCoordinates(:,2);
tC = cegaCoordinates(:,3);
% ground truth coordinates
xT = TrueCoordinates(:,1);
yT = TrueCoordinates(:,2);
tT = TrueCoordinates(:,3);
% spot finder coordinates
xS = BgSpCoordinates(:,1);
yS = BgSpCoordinates(:,2);
tS = BgSpCoordinates(:,3);

imsz = size(im);
% get cross bounds for CEGA
xCLow = xC-1;
xCHigh = xC+1;
yCLow = yC-1;
yCHigh = yC+1;
xCLow(xCLow<1) = 1;
yCLow(yCLow<1) = 1;
xCHigh(xCHigh>imsz(1)) = imsz(1);
yCHigh(yCHigh>imsz(2)) = imsz(2);
% get cross bounds for true coordinates
xTLow = xT-1;
xTHigh = xT+1;
yTLow = yT-1;
yTHigh = yT+1;
xTLow(xTLow<1) = 1;
yTLow(yTLow<1) = 1;
xTHigh(xTHigh>imsz(1)) = imsz(1);
yTHigh(yTHigh>imsz(2)) = imsz(2);
% get cross bounds for spot coordinates
xSLow = xS-1;
xSHigh = xS+1;
ySLow = yS-1;
ySHigh = yS+1;
xSLow(xSLow<1) = 1;
ySLow(ySLow<1) = 1;
xSHigh(xSHigh>imsz(1)) = imsz(1);
ySHigh(ySHigh>imsz(2)) = imsz(2);


% for built in implay
% make an rgb tensor
% let's try the KL movie for shits and giggles
%outIm = repmat(filterMovies.KLM,[1 1 1 3]);
outIm = repmat(im,[1 1 1 3]);
outIm = permute(outIm,[1 2 4 3]);
outIm = outIm/max(outIm(:))*255; % rescale to 8 bit
% zero out x's on implay movie for CEGA
for ii = 1:length(xC)
   outIm(xCLow(ii):xCHigh(ii),yC(ii),:,tC(ii)) = 0;
   outIm(xC(ii),yCLow(ii):yCHigh(ii),:,tC(ii)) = 0;
end
% zero out x's on implay movie for true positions
for ii = 1:length(xT)
   outIm(xTLow(ii):xTHigh(ii),yT(ii),:,tT(ii)) = 0;
   outIm(xT(ii),yTLow(ii):yTHigh(ii),:,tT(ii)) = 0;
end
% zero out x's on implay movie for spot positions
for ii = 1:length(xS)
   outIm(xSLow(ii):xSHigh(ii),yS(ii),:,tS(ii)) = 0;
   outIm(xS(ii),ySLow(ii):ySHigh(ii),:,tS(ii)) = 0;
end
% draw red x's on implay movie for cega
for ii = 1:length(xC)
   outIm(xCLow(ii):xCHigh(ii),yC(ii),1,tC(ii)) = 255;
   outIm(xC(ii),yCLow(ii):yCHigh(ii),1,tC(ii)) = 255;
end
% draw green x's on implay movie for true positions
for ii = 1:length(xT)
   outIm(xTLow(ii):xTHigh(ii),yT(ii),2,tT(ii)) = 255;
   outIm(xT(ii),yTLow(ii):yTHigh(ii),2,tT(ii)) = 255;
end
% draw blue x's on implay movie for spot positions
for ii = 1:length(xS)
   outIm(xSLow(ii):xSHigh(ii),yS(ii),3,tS(ii)) = 255;
   outIm(xS(ii),ySLow(ii):ySHigh(ii),3,tS(ii)) = 255;
end

% convert to uint8 and run implay
implay(uint8(outIm));

