% scoreCega.m
% script to score box finding on simulated data with and without Cega filter
% Author: PKR, August 2020
% License: GPL-3.0-or-Later

%% Set Gain and Offset values, assuming read noise is negligible for exposition
Gain = 70;
Offset = 2293;

%% Call motor sim class
T = motorSim;
% set parameters
T.gain = Gain;
T.offset = Offset;
T.MeanPhotons = 250;
T.N = [15,15];
% generate simulation
T.buildDataSet;
TrueCoordinates = T.SimLocalizations;
% score Cega against the movie
sigmas = [1 1.5];
% may have to update cega params, use defaults for now
[CegaCoordinates,CegaMovies] = cega(T.SimMovie,0.5,31,sigmas,1);
SpotCoordinates = findSpots(T.SimMovie,sigmas,5);
%% get time indices for all coordinates
CegaTimes = CegaCoordinates(:,3);
SpotTimes = SpotCoordinates(:,3);
TrueTimes = TrueCoordinates(:,3);

%% perform JIndex scoring
CegaTP = 0;
CegaFP = 0;
CegaFN = 0;
SpotTP = 0;
SpotFP = 0;
SpotFN = 0;

% loop over all frames perform Tracking problem between coordinates
for ii = 1:T.sz(3)
    TrueInd = TrueTimes==ii;
    CegaInd = CegaTimes==ii;
    SpotInd = SpotTimes==ii;

    sT = sum(TrueInd);
    sC = sum(CegaInd);
    sS = sum(SpotInd);

    FalseNegative = diag(4*ones(sT,1));
    FalsePositiveCega = diag(4*ones(sC,1));
    FalsePositiveSpot = diag(4*ones(sS,1));

    TrueX = TrueCoordinates(TrueInd,1);
    TrueY = TrueCoordinates(TrueInd,2);
    CegaX = CegaCoordinates(CegaInd,1);
    CegaY = CegaCoordinates(CegaInd,2);
    SpotX = SpotCoordinates(SpotInd,1);
    SpotY = SpotCoordinates(SpotInd,2);

    CegaCost = zeros(sC,sT);
    SpotCost = zeros(sS,sT);

    for jj = 1:sT
        CegaCost(:,jj) = (TrueX(jj)-CegaX).^2+(TrueY(jj)-CegaY).^2+1;
        SpotCost(:,jj) = (TrueX(jj)-SpotX).^2+(TrueY(jj)-SpotY).^2+1;
    end
    CegaCost(CegaCost>4) = 0;
    SpotCost(SpotCost>4) = 0;
    % get lower right hand corner matrices
    CegaJM = (CegaCost>0)'*eps;
    SpotJM = (SpotCost>0)'*eps;

    % I think I screwed up falsepositive and negative, need to verify
    CegaFrame = [CegaCost FalsePositiveCega; FalseNegative CegaJM];
    SpotFrame = [SpotCost FalsePositiveSpot; FalseNegative SpotJM];

    [Cega2TrueLinks, True2CegaLinks, cegaCost] = lap(CegaFrame);
    [Spot2TrueLinks, True2SpotLinks, spotCost] = lap(SpotFrame);
    
    % start building Jaccard Index scoring
    % get TP from Cega2TrueLinks for all vectors 1:sC <= sT
    CegaTP = CegaTP + sum(Cega2TrueLinks(1:sC) <= sT);
    SpotTP = SpotTP + sum(Spot2TrueLinks(1:sS) <= sT);
    % get FP from Cega2TrueLinks for all vectors 1:sC > sT
    CegaFP = CegaFP + sum(Cega2TrueLinks(1:sC) > sT);
    SpotFP = SpotFP + sum(Spot2TrueLinks(1:sS) > sT);
    % get FN from True2CegaLinks for all vectors 1:sT > sC
    CegaFN = CegaFN + sum(True2CegaLinks(1:sT) > sC);
    SpotFN = SpotFN + sum(True2SpotLinks(1:sT) > sS);
end

CegaJIndex = CegaTP/(CegaTP+CegaFP+CegaFN);
SpotJIndex = SpotTP/(SpotTP+SpotFP+SpotFN);
