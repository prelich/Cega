% scoreCega.m
% script to score box finding on simulated data with and without Cega filter
% Author: PKR, August 2020
% License: GPL-3.0-or-Later

%% set randon number generator seed
rng(8675309);

%% Set Gain and Offset values, assuming read noise is negligible for exposition
Gain = 70;
Offset = 2293;
%% Set vector of mean photon counts for simulation studies
MeanPhotons = [50 100 150 200 250 300 350 400 450 500 550 600];
%% Call motor sim class
T = motorSim;
% set constant sim parameters
T.gain = Gain;
T.offset = Offset;
T.N = [15,15];
%% initialize output variables
numRuns = length(MeanPhotons);
CegaJIndex = zeros(numRuns,1);
SpotJIndex = zeros(numRuns,1);
CegaRecovery = zeros(numRuns,1);
SpotRecovery = zeros(numRuns,1);
CegaTP = zeros(numRuns,1);
CegaFP = zeros(numRuns,1);
CegaFN = zeros(numRuns,1);
SpotTP = zeros(numRuns,1);
SpotFP = zeros(numRuns,1);
SpotFN = zeros(numRuns,1);

%KLThreshes = [0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5 10.5 11.5];
KLThreshes = [1 1.5 3 5 7 9 10 10 20 20 20 20];
% 0.01 connThresh and 2 KL had decent recovery rates for 150 photons
% crappy Jaccard index though
% scale spaces for cega and find spots
sigmas = [1 1.5];

%% loop over range of input photon values
for nn = 1:numRuns
    T.MeanPhotons = MeanPhotons(nn);
    % generate simulation
    T.buildDataSet;
    TrueCoordinates = T.SimLocalizations;
    % score Cega against the movie
    % cega inputs: movie, ConnThresh, WindowLength, sigmas, KLThreshold
    [CegaCoordinates,CegaMovies] = cega(T.SimMovie,0.01,31,sigmas,KLThreshes(nn));
    % spot inputs: movie, sigmas, pixel threshold
    % determine average intensity at center of psf with erf function
    avIntensity = T.MeanPhotons*((erf((0.5)/sigmas(1)/sqrt(2))...
        -erf((-0.5)/sigmas(1)/sqrt(2)))/2)^2;
    % threshold from 34% quantile of normal distribution at avIntensity
    pxThresh = avIntensity + sqrt(avIntensity*2)*erfinv(2*0.48-1); 
    [SpotCoordinates, LoGs] = findSpots(T.SimMovie,sigmas,pxThresh);
    %% get time indices for all coordinates
    CegaTimes = CegaCoordinates(:,3);
    SpotTimes = SpotCoordinates(:,3);
    TrueTimes = TrueCoordinates(:,3);

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

        [Cega2TrueLinks, True2CegaLinks, cegaCost] = lap(CegaFrame,0,0,0);
        [Spot2TrueLinks, True2SpotLinks, spotCost] = lap(SpotFrame,0,0,0);
    
        % start building Jaccard Index scoring
        % get TP from Cega2TrueLinks for all vectors 1:sC <= sT
        CegaTP(nn) = CegaTP(nn) + sum(Cega2TrueLinks(1:sC) <= sT);
        SpotTP(nn) = SpotTP(nn) + sum(Spot2TrueLinks(1:sS) <= sT);
        % get FP from Cega2TrueLinks for all vectors 1:sC > sT
        CegaFP(nn) = CegaFP(nn) + sum(Cega2TrueLinks(1:sC) > sT);
        SpotFP(nn) = SpotFP(nn) + sum(Spot2TrueLinks(1:sS) > sT);
        % get FN from True2CegaLinks for all vectors 1:sT > sC
        CegaFN(nn) = CegaFN(nn) + sum(True2CegaLinks(1:sT) > sC);
        SpotFN(nn) = SpotFN(nn) + sum(True2SpotLinks(1:sT) > sS);
    end

    CegaJIndex(nn) = CegaTP(nn)/(CegaTP(nn)+CegaFP(nn)+CegaFN(nn));
    SpotJIndex(nn) = SpotTP(nn)/(SpotTP(nn)+SpotFP(nn)+SpotFN(nn));

    CegaRecovery(nn) = CegaTP(nn)/(CegaTP(nn)+CegaFN(nn));
    SpotRecovery(nn) = SpotTP(nn)/(SpotTP(nn)+SpotFN(nn));
end

%% plot the results
figure;
bar([CegaJIndex,SpotJIndex]);
title('Jaccard Index of Cega and Standard Blob Finder');
xlabel('Average Probe Intensity (Photons)');
xticklabels(MeanPhotons);
ylabel('Jaccard Index Score');
legend('Cega','Standard');


figure;
bar([CegaRecovery,SpotRecovery]);
title('Recovery Rate of Cega and Standard Blob Finder');
xlabel('Average Probe Intensity (Photons)');
ylabel('Recovery Rate');
xticklabels(MeanPhotons);
legend('Cega','Standard');
