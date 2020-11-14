classdef motorSim < handle
    % Molecular Motor Fluorescence Simulation
    % Author: PKR, July 2020
    % NOTE: MATLAB seed command is global, declare it outside of class

    properties
        % output data structures
        bg;
        im;
        sz;
        SimTracks;
        SimLocalizations;
        SimMovie;
        % intermediate data structures
        nN;
        NTracks;
        Nactive;
        emissionTimes;
        motorPhotons;
        % experimental parameters
        gain = 70; % effective photon gain (ADU/e-)
        offset = 2293; % ADU value added to image
        WinLength = 7; % temporal window for median filter, must be odd
        % simulation parameters
        N = [15,15]; % number of motors for each pop. on the image at any frame
        D = [0.5,0.5]; % diffusion rate by population, isotropic
        V = [0.63,0.59]; % mean speed of each motor population
        VMax = [2.0,2.0]; % max speed of motor pop. (confined max)
        InitSigV = [0.5,0.5]; % lognormal sigma of pop. mean speed
        TimeSigV = [0.01,0.01]; % normal sigma of the change in pop speed
        NumPops; % population number, calculated when building rough tracks
        % photon painting parameters
        PSFSigma = 0.75; % psf sigma in pixels
        MeanPhotons = 250; % photons per motor per frame
        DBridge = 0.5; % diffusion constant of brownian bridge interpolator
        WhiteNoiseSigma = 0.1; % sigma for white noise in image
    end

    methods
        %% constructor
        function obj = motorSim(fileLoc)
            if nargin > 0
                % this doesn't apply gain settings
                obj.im = loadTifFile(fileLoc);
                obj.sz = size(obj.im);
            end
        end
        %% run everything, generate a simulation
        function obj = buildDataSet(obj)
            if isempty(obj.im)
                obj.loadExperiment;
                obj.prepBackground;
            end
            if isempty(obj.bg)
                obj.prepBackground;
            end
            obj.getRoughTracks;
            obj.getEmissionTimes;
            obj.paintSim;
        end

        function obj = loadExperiment(obj)
            [fileName,pathLoc] = uigetfile('.tif','Select a tif stack to analyze');
            fileLoc = fullfile(pathLoc,fileName);
            % load the image
            obj.loadTiffFile(fileLoc);
            % apply gain settings
            obj.applyGain;
            obj.sz = size(obj.im);
        end

        function obj = prepBackground(obj)
            % create background movie with temporal filter
            HalfWin = floor(obj.WinLength/2);
            obj.bg = 0*obj.im;
            for ii = 1:HalfWin
                obj.bg(:,:,ii) = median(obj.im(:,:,1:ii+HalfWin),3);
            end
            for ii = HalfWin+1:obj.sz(3)-HalfWin-1
                obj.bg(:,:,ii) = median(obj.im(:,:,ii-HalfWin:ii+HalfWin),3);
            end
            for ii = obj.sz(3)-HalfWin:obj.sz(3)
                obj.bg(:,:,ii) = median(obj.im(:,:,ii-HalfWin:end),3);
            end
        end

        function obj = getRoughTracks(obj)
            obj.NumPops = length(obj.N); % get population number here
            %% generate a set of tracks for the moving motors
            % when a motor leaves the ROI, kill the track and start a new one
            %numFrames = obj.sz(3);
            yMax = obj.sz(2); % get ROI boundaries, setting x axis = 1, y axis = 2
            xMax = obj.sz(1);
            %% pick starting positions and velocities of motors
            NStarts = cell(obj.NumPops,1);
            vNStarts = cell(obj.NumPops,1);
            for ii = 1:obj.NumPops
                NStarts{ii} = rand(obj.N(ii),2).*repmat([xMax,yMax],obj.N(ii),1);
                vNStarts{ii} = lognrnd(log(obj.V(ii)),obj.InitSigV(ii),obj.N(ii),2);
                vNStarts{ii}(NStarts{ii}(:,1) > xMax/2,1) = ...
                    -vNStarts{ii}(NStarts{ii}(:,1) > xMax/2,1);
                vNStarts{ii}(NStarts{ii}(:,2) > yMax/2,2) = ...
                    -vNStarts{ii}(NStarts{ii}(:,2) > yMax/2,2);
            end
            %% track trajectories, measured before and after each frame
            obj.NTracks = cell(obj.NumPops,1);
            obj.Nactive = cell(obj.NumPops,1);
            % Start initial track structures
            % Tracks have variable length so use cell array of matrices
            for ii = 1:obj.NumPops
                tempTracks = {};
                for jj = 1:obj.N(ii)
                    tempTracks{jj} = [NStarts{ii}(jj,:),0,vNStarts{ii}(jj,:)];
                end
                obj.NTracks{ii} = tempTracks;
                obj.Nactive{ii}{1} = 1:obj.N(ii);
            end
            % loop over entire movie length and generate tracks
            obj.nN = obj.N+1; % next track element for each population
            for ii = 1:obj.NumPops
                for tt = 1:obj.sz(3)
                    obj.Nactive{ii}{tt+1} = [];
                    for elem = obj.Nactive{ii}{tt}
                        nCoordinates = obj.NTracks{ii}{elem}(end,:);
                        % drift w/ isotropic diffusion for position
                        nCoordinates(1) = nCoordinates(1) + nCoordinates(4) + ...
                            sqrt(2*obj.D(ii))*randn();
                        nCoordinates(2) = nCoordinates(2) + nCoordinates(5) + ...
                            sqrt(2*obj.D(ii))*randn();
                        nCoordinates(3) = tt;
                        % isotropic confined diffusion for velocity
                        nCoordinates(4) = nCoordinates(4)+obj.TimeSigV(ii)*randn();
                        nCoordinates(4) = max(-obj.VMax(ii),min(obj.VMax(ii),nCoordinates(4)));
                        nCoordinates(5) = nCoordinates(5)+obj.TimeSigV(ii)*randn();
                        nCoordinates(5) = max(-obj.VMax(ii),min(obj.VMax(ii),nCoordinates(5)));
                        obj.NTracks{ii}{elem} = [obj.NTracks{ii}{elem}; nCoordinates];
                        % kill a track if it wanders out of ROI, start a new track
                        if (nCoordinates(1) < 0 || nCoordinates(1) > obj.sz(1) || ...
                            nCoordinates(2) < 0 || nCoordinates(2) > obj.sz(2))
                            obj.Nactive{ii}{tt+1} = [obj.Nactive{ii}{tt+1}, obj.nN(ii)];
                            newNStart = rand(1,2).*[xMax,yMax];
                            newNVelocity = lognrnd(log(obj.V(ii)),obj.InitSigV(ii),1,2);
                            newNVelocity(newNStart(1) > xMax/2,1) = ...
                                -newNVelocity(newNStart(1) > xMax/2,1);
                            newNVelocity(newNStart(2) > yMax/2,2) = ...
                                -newNVelocity(newNStart(2) > yMax/2,2);
                            obj.NTracks{ii}{obj.nN(ii)} = [newNStart, tt, newNVelocity];
                            obj.nN(ii) = obj.nN(ii)+1;
                        else
                            obj.Nactive{ii}{tt+1} = [obj.Nactive{ii}{tt+1}, elem];
                        end
                    end
                end
            end
        end

        function obj = getEmissionTimes(obj)
            %% get emission times of photons for each blob
            obj.emissionTimes = cell(obj.NumPops,1);
            for ii = 1:obj.NumPops
                emissionPop = {};
                for mm = 1:obj.nN(ii)-1
                    startTime = obj.NTracks{ii}{mm}(1,3);
                    endTime = obj.NTracks{ii}{mm}(end,3);
                    TimeGap = endTime-startTime;
                    sampleSize = ceil(obj.MeanPhotons*TimeGap + 5*sqrt(obj.MeanPhotons*TimeGap));
                    timeCDFs = rand(sampleSize,1);
                    % convert CDFs to emission times
                    xx = -log(1-timeCDFs)/obj.MeanPhotons;
                    xx = cumsum(xx)+startTime;
                    xx = xx(xx<endTime);
                    emissionPop{mm} = xx;
                end
                obj.emissionTimes{ii} = emissionPop;
            end
        end

        function obj = paintSim(obj)
            %% paint photons from tracks and emission time vectors
            %% output a tracks array, localization list, and photon movie
            obj.motorPhotons = zeros(obj.sz);
            obj.SimTracks = cell(obj.NumPops,1);
            obj.SimLocalizations = [];
            for ii = 1:obj.NumPops
                obj.SimTracks{ii} = {};
                for mm = 1:obj.nN(ii)-1 % last nN is never used
                    obj.SimTracks{ii}{mm} = [];
                    cc = 1; % counter for emission times
                    trackStartTime = floor(obj.NTracks{ii}{mm}(1,3));
                    numEmissions = length(obj.emissionTimes{ii}{mm});
                    PhotonDeviations = obj.PSFSigma*randn(numEmissions,2);
                    BBridgeDraws = randn(numEmissions,2);
                    while cc < numEmissions
                        startEmission = cc;
                        tt = ceil(obj.emissionTimes{ii}{mm}(cc));
                        flag = true;
                        while flag
                            cc=cc+1;
                            try
                                flag = ceil(obj.emissionTimes{ii}{mm}(cc))==tt;
                            catch ME
                                flag = false;
                            end
                        end
                        startPos = obj.NTracks{ii}{mm}(tt-trackStartTime,1:2);
                        endPos =  obj.NTracks{ii}{mm}(tt-trackStartTime+1,1:2);
                        % get time gaps between emission events 
                        emitFrame = obj.emissionTimes{ii}{mm}(startEmission:cc-1)-tt+1;
                        dtau = diff([0; emitFrame]);
                        endTau = 1 - emitFrame(end);
                        % create brownian bridge deviations
                        BBridgeDeviations = repmat(sqrt(dtau*2*obj.DBridge),1,2).* ...
                            BBridgeDraws(startEmission:cc-1,:);
                        particlePosition = cumsum(BBridgeDeviations)+startPos;
                        % sample a jump to the end of the frame
                        maxDisplacement = sqrt(endTau*2*obj.DBridge)*randn(1,2)+ ...
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
                        photonPosition = photonPosition(photonPosition(:,1)<obj.sz(1),:);
                        photonPosition = photonPosition(photonPosition(:,2)<obj.sz(2),:);
                        frameCoordinates = [mean(photonPosition,1),tt,length(photonPosition)];
                        % put photon averages into simulated tracks and localization structures
                        obj.SimLocalizations = [obj.SimLocalizations; frameCoordinates];
                        obj.SimTracks{ii}{mm} = [obj.SimTracks{ii}{mm}; frameCoordinates];
                        % paint the motorPhotons movie
                        pixelPosition = ceil(photonPosition);
                        for jj = 1:size(photonPosition,1)
                            obj.motorPhotons(pixelPosition(jj,1),pixelPosition(jj,2),tt) = ...
                            obj.motorPhotons(pixelPosition(jj,1),pixelPosition(jj,2),tt) + 1;
                        end
                    end
                end
            end

            % create poisson-like realization of the background
            bgRealization = poissrnd(obj.bg);
            obj.SimMovie = obj.WhiteNoiseSigma*randn(obj.sz)+bgRealization+obj.motorPhotons;
        end

        function obj = applyGain(obj) 
            obj.im = (obj.im-obj.offset)/obj.gain;
            obj.im(obj.im <= 0) = 0.01; % no negative numbers
        end

        function obj = loadTiffFile(obj,fileLoc)
            % Load the tif file`
            InfoImage=imfinfo(fileLoc);
            mImage=InfoImage(1).Width;
            nImage=InfoImage(1).Height;
            NumberImages=length(InfoImage);
            obj.im=zeros(nImage,mImage,NumberImages,'uint16');
            for i=1:NumberImages
                obj.im(:,:,i)=imread(fileLoc,'Index',i);
            end
            obj.im = single(obj.im); % change data type
        end

    end
end
