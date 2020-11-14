% cegaBackgroundTest.m
% script to determine KL divergence values to do poisson noise from
% background simulation
% Author: PKR, August 2020
% License: GPL-3.0-or-Later

%% set randon number generator seed
rng(8675309);

%% Set Gain and Offset values, assuming read noise is negligible for exposition
Gain = 70;
Offset = 2293;

%% Call motor sim class
T = motorSim;
% set constant sim parameters
T.gain = Gain;
T.offset = Offset;
T.WinLength = 7;
T.N = [15,15];
% no probes, lets just look at the background
T.MeanPhotons = 0;
% generate simulation
T.buildDataSet;
TrueCoordinates = T.SimLocalizations;
% score Cega against the movie
sigmas = [1 1.5];
% may have to update cega params, use defaults for now
% cega inputs: movie, ConnThresh, WindowLength, sigmas, KLThreshold
[CegaCoordinates,CegaMovies] = cega(T.SimMovie,0.01,31,sigmas,0.1);

Ints = T.SimMovie(:);
KL = CegaMovies.KLM(:);

figure;
subplot(2,1,1);
histogram(Ints);
title('Histogram of Background Simulation Intensities All Pixels');
xlabel('Intensity Values');
ylabel('Counts');
subplot(2,1,2);
histogram(log(KL+eps));
title('Histogram of KL Divergence of Background Simulation All Pixels');
xlabel('Log KL Divergence Values');
ylabel('Counts');
