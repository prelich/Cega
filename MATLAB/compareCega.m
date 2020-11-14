% compareCega.m
% script to compare Cega filter with our tracker filter on sample data
% Author: PKR, July 2020
% License: GPL-3.0-or-Later

%% We'll hard code a path with the sample file for publication!
% fileLoc = ('path_to_file/filename.tif');
%% Use a prompt if user has custom data
[fileName,pathLoc] = uigetfile('.tif','Select a tif stack to analyze');
fileLoc = fullfile(pathLoc,fileName);

%% Load the tif file
InfoImage=imfinfo(fileLoc);
mImage=InfoImage(1).Width;
nImage=InfoImage(1).Height;
NumberImages=length(InfoImage);
Vector=zeros(nImage,mImage,NumberImages,'uint16');
for i=1:NumberImages
   Vector(:,:,i)=imread(fileLoc,'Index',i);
end

%% Set Gain and Offset values, assuming read noise is negligible for exposition
Gain = 70;
Offset = 2293;
%% Use a prompt if user has custom data
% Gain = inputdlg('What is the Camera gain for this movie? (ADU/e-)');
% Offset = inputdlg('What is the Camera offset for this movie? (ADU)');
%% Tansform movie units from ADU to effective photons!
im = (single(Vector)-Offset)/Gain; % make sure to convert tiffs from uint to single format!
%% Run CEGA method, return filtered movies and maxima
ConnThresh = 0.5; % Connectivity Threshold
WinLength = 31; % Median Filter Window Length
sigmas = [1, 1.5]; % sigmas for scale space filtering
KLThresh = 0.75; % KL Divergence Threshold for selecting maxima
[cegaCoordinates,filterMovies] = cega( im, ConnThresh, WinLength, sigmas, KLThresh )
%% Run getBoxesT method, return filtered movies and maxima
[ResultsFilterMaxima,filterMoviesB] = getBoxesT( im, 1, sigmas, KLThresh, false, ConnThresh, WinLength);
boxCoordinates = ResultsFilterMaxima.maxima;
%% Paint coordinates onto the final image to see how identification looks
xC = cegaCoordinates(:,1);
yC = cegaCoordinates(:,2);
tC = cegaCoordinates(:,3);

xB = boxCoordinates(:,1);
yB = boxCoordinates(:,2);
tB = boxCoordinates(:,3);

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
% get cross bounds for box filter
xBLow = xB-1;
xBHigh = xB+1;
yBLow = yB-1;
yBHigh = yB+1;
xBLow(xBLow<1) = 1;
yBLow(yBLow<1) = 1;
xBHigh(xBHigh>imsz(1)) = imsz(1);
yBHigh(yBHigh>imsz(2)) = imsz(2);

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
% zero out x's on implay movie for box filter
for ii = 1:length(xB)
   outIm(xBLow(ii):xBHigh(ii),yB(ii),:,tB(ii)) = 0;
   outIm(xB(ii),yBLow(ii):yBHigh(ii),:,tB(ii)) = 0;
end
% draw red x's on implay movie for cega
for ii = 1:length(xC)
   outIm(xCLow(ii):xCHigh(ii),yC(ii),1,tC(ii)) = 255;
   outIm(xC(ii),yCLow(ii):yCHigh(ii),1,tC(ii)) = 255;
end
% draw green x's on implay movie for box filter
for ii = 1:length(xB)
   outIm(xBLow(ii):xBHigh(ii),yB(ii),2,tB(ii)) = 255;
   outIm(xB(ii),yBLow(ii):yBHigh(ii),2,tB(ii)) = 255;
end
% convert to uint8 and run implay
implay(uint8(outIm));

