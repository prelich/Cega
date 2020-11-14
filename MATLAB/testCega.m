% testCega.m
% script to test Cega filter on sample data
% Author: PKR, June 2020
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
%% Run CEGA method, return filtered movies
ConnThresh = 0.5; % Connectivity Threshold
WinLength = 31; % Median Filter Window Length
sigmas = [1, 1.5]; % sigmas for scale space filtering
KLThresh = 0.75; % KL Divergence Threshold for selecting maxima
[rawCoordinates,filterMovies] = cega( im, ConnThresh, WinLength, sigmas, KLThresh );
%rawCoordinates = findSpots(im+40,sigmas,8);
%% Paint coordinates onto the final image to see how identification looks
x = rawCoordinates(:,1);
y = rawCoordinates(:,2);
t = rawCoordinates(:,3);

imsz = size(im);
xLow = x-1;
xHigh = x+1;
yLow = y-1;
yHigh = y+1;
xLow(xLow<1) = 1;
yLow(yLow<1) = 1;
xHigh(xHigh>imsz(1)) = imsz(1);
yHigh(yHigh>imsz(2)) = imsz(2);

% for built in implay
% make an rgb tensor
%outIm = repmat(filterMovies.KLM,[1 1 1 3]);
outIm = repmat(im,[1 1 1 3]);
outIm = permute(outIm,[1 2 4 3]);
outIm = outIm/max(outIm(:))*255; % rescale to 8 bit
% draw red x's on implay movie
for ii = 1:length(x)
   outIm(xLow(ii):xHigh(ii),y(ii),:,t(ii)) = 0;
   outIm(xLow(ii):xHigh(ii),y(ii),1,t(ii)) = 255;
   outIm(x(ii),yLow(ii):yHigh(ii),:,t(ii)) = 0;
   outIm(x(ii),yLow(ii):yHigh(ii),1,t(ii)) = 255;
end
% convert to uint8 and run implay
implay(uint8(outIm));

