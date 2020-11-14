function [x, y, Assignment_cost] = lap(cc, NONLINK_MARKER, extendedTesting, augmentCC, noLinkCost)
%% PKR NOTE: This file was taken from U-Track 2012 and then modified to use
%% built in C functions so I can efficiently see what costs were chosen.
%LAP solves the linear assignment problem for a given cost matrix
%
% A linear assignment tries to establish links between points in two sets.
% One point in set A can only link to one point in set B and vice versa.
% The cost associated with the link from element i of A to element j of B
% is given by cc(i,j).
%
% SYNOPSIS [x, y, Assignment_cost] = lap(cc, NONLINK_MARKER, extendedTesting, augmentCC, noLinkCost)
%
% INPUT:  cc: cost matrix, which has to be square. Set cost(i,j) to the
%             value of NONLINK_MARKER, if the link is not allowed. cc can
%             be input as a sparse matrix (in which case there won't be any
%             NONLINK_MARKERs).
%             Elements of the cost matrix cannot be inf or nan!
%
%             For an unequal number of points (or, generally, if birth and
%             death is to be allowed) the cost matrix is formed as
%             a 2-by-2 catenation of four sub-matrices. If there are n
%             elements in set A and m elements in set B, sub-matrix (1,1)
%             is an n-by-m matrix with the cost for each link. The two
%             off-diagonal sub-matrices make non-links possible. They are
%             both diagonal square matrices (2,1: m-by-m; 1,2: n-by-n) with
%             the cost for not linking a point (e.g. determined by a
%             maximum search radius) in the diagonal and NONLINK_MARKER off
%             the diagonal. Sub-matrix (2,2) is a m-b-n matrix of any
%             not-NONLINK_MARKER value (suggested: one, except for sparse
%             matrix input)
%
% NONLINK_MARKER : value to indicate that two points cannot be linked.
%             Default: -1. NaN is not allowed here.
%
% extendedTesting (used to be optional, now input has no effect and
%                  extendedTesting will always be performed)
%                LAP will always make sure that:
%                  - There cannot be NaNs in cc
%                  - In every row and every column of cc there must be
%                    at least 1 element that is not a NONLINK_MARKER.
%
% augmentCC  (optional, [{0}/1]) If 1, the cost matrix will be augmented to
%            allow births and deaths.
% noLinkCost (optional, [{maximum cost + 1}]) Cost for linking a feature
%            to nothing.
%
%
% OUTPUT: x: The point A(i) links to B(x(i))
%         y: The point B(j) links to A(y(j))
%       Assignment_cost: the cost of this set of assignments
%
%            Any x > m or y > n indicates that this point is not linked.
%
% EXAMPLE
%
% % Make a set of points p1 and slightly shift the points in p2
% p1 = rand(5,2)*10;
% p2 = p1+rand(5,2);
% % plot the data
% figure,plot(p1(:,1),p1(:,2),'.r',p2(:,1),p2(:,2),'.b')
% % invert the order of p2 (to make it a bit more interesting) and remove the
% % third point
% p2 = p2(end:-1:1,:);
% p2(3,:) = [];
% % calculate distance matrix between the point sets
% dm = distMat2(p1,p2);
% % remove all links that are longer than a threshold (mark with -1)
% dm(dm>5) = -1;
% % print distance matrix
% disp(dm)
% % link via LAP. nonlinkmarker=-1, augment the matrix and show
% [links12, links21] = lap(dm,-1,0,1)
% % links12 should now look something like
% % links12 =
% %
% %            4
% %            3
% %            7
% %            2
% %            1
% %            9
% %            5
% %            6
% %            8
% %
% % For links12, the first five entries are relevant (because p1 consists of
% % 5 points). The first point in p1 links to point #4 in set 2
% % (link12(1)==4). The third point in p1 links nowhere - there are only 4
% % points in links21, so an index above 4 means that a link to a virtual
% % point had to be made.
% % Links21 is analogous, except that only the first four entries count.
%
%
% lapjv is implemented through a MEX DLL.
%
%    u-track is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
%
%    u-track is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with u-track.  If not, see <http://www.gnu.org/licenses/>.
%
%
% Copyright: Ge Yang, LCCB, TSRI, Dec. 10, 2004
% extended by jonas 6/05


% this function automatically extracts the sparse matrix
% representation of cc.

% int *col = new int[size * NEIGHBOR_NUM_MAX + 1];
% int *first = new int[size + 2];
% int *x = new int[size + 1];
% int *y = new int[size + 1];
% double *u = new double[size + 1];
% double *v = new double[size + 1];
% double  *cc = new double[size * NEIGHBOR_NUM_MAX + 1];

% MODIFIED BY PKR TO INCLUDE ASSIGNMENT COST, UNM, 7/11/2013

%=====================
% TEST INPUT
%=====================

if (nargin == 1) || isempty(NONLINK_MARKER)
    NONLINK_MARKER = -1;
elseif isnan(NONLINK_MARKER)
    error('NONLINK_MARKER cannot be NaN!')
end

if nargin < 4 || isempty(augmentCC)
    augmentCC = 0;
end

% test size
scc = size(cc);
% check size only if no augmentation
if ~augmentCC
    if scc(1) ~= scc(2) || length(scc) > 2
        error('cost must be a 2D square matrix!')
    end
elseif length(scc) > 2
    error('cost must be a 2D matrix!')
else
    % if we're augmenting, sparse matrices are produced. This will be
    % problematic with cost 0
    if any(cc(:)==0) && ~issparse(cc)
        validCC = cc ~= NONLINK_MARKER;
        if any(any(cc(validCC))) < 0
            warning('LAP:negativeCosts',...
                'there are negative and zero costs. This could lead to errors')
        end
        cc(validCC) = cc(validCC) + 10;
    end
end

if nargin < 5 || isempty(noLinkCost)
    noLinkCost = max(max(cc)) + 1;
end

% if we have -1 as non-link-marker, and we get it everywhere, we get
% a segmentation fault.
if noLinkCost == NONLINK_MARKER
    noLinkCost = noLinkCost + 1;
    if noLinkCost == 0
        noLinkCost = 0.01;
    end
end

%=================================
% AUGMENT COST MATRIX IF SELECTED
%=================================
if augmentCC
    % check if sparse
    if issparse(cc)
        costLR = cc';
        cc = [cc, spdiags(noLinkCost * ones(scc(1),1), 0, scc(1), scc(1));...
            spdiags(noLinkCost * ones(scc(2),1), 0, scc(2), scc(2)),...
            costLR];
    else
        % make cc sparse. Take NLM in cc into account!
        cc(cc==NONLINK_MARKER) = 0;
        cc=sparse(cc);
        costLR = cc';
        cc = [cc, diag(noLinkCost * ones(scc(1),1)); ...
            diag(noLinkCost * ones(scc(2),1)), ...
            costLR];
    end
    % remember that the size of the matrix has increased!
    scc = [sum(scc), sum(scc)];
    clear costLR colIdx rowIdx
end

%=============================
% CALCULATE SPARSE MATRIX
%=============================
% Calculate sparse representation of cost matrix with compactCC (all
% significant elements of cc), fst and kk. Read cc in rows, not cols!
% fst: for every first significant element of a row: index into compactCC
% kk : for every significant element: column
% do the work on the transposed cost matrix!
cc = cc';
% find the significant elements. If sparse input, find nonzero elements
if issparse(cc)
    [rowIdx, colIdx, val] = find(cc);
else
    [rowIdx, colIdx] = find(cc ~= NONLINK_MARKER);
    val = cc(cc ~= NONLINK_MARKER);
end
    % test that all cols and all rows are filled, and that there are no nans
    if issparse(cc)
        %         if (~all(sum(cc,1)) || ~all(sum(cc,2)))
        if (~all(sum(logical(cc),1)) || ~all(sum(logical(cc),2))) %updated by Pat Cutler January 19, 2011 so that negatives don't cancel out values
            error('LAP:BadCostMatrix',...
                'Rows and columns of the cost matrix must allow at least one possible link!')
        end
    else
        if (~all(sum(cc~=NONLINK_MARKER,1)) || ~all(sum(cc~=NONLINK_MARKER,2)))
            error('LAP:BadCostMatrix',...
                'Rows and columns of the cost matrix must allow at least one possible link!')
        end
    end
    if any(~isfinite(val))
        error('LAP:NanCostMatrix','Cost matrix cannot contain NaNs or Inf!')
    end

clear cc;
rowIdx = int32([0;rowIdx]);
% PKR:added colIdx conversion, I need to reference val again later to
% output costs.
colIdx = int32([0;colIdx]);
%colIdx = int32([0;find(diff([0;colIdx]));length(val)]);
val = double([0; val]);

%==================================
% CALL MEX-FUNCTION
%==================================

%PKR - the costs associated with u and v seem to be massive, I think the
%information is lost by the way mexLap processes things...
[x, y, u, v] = ...
    mexLap(double(scc(1)), int32(length(val)), val, ...
    rowIdx, int32([0;find(diff(colIdx));length(val)])); %#ok<NASGU>

%==================================

% remove first element from output vectors, as it is a meaningless 0.
x = x(2:end);
y = y(2:end);
% modified ismember to be a lot faster, less overhead for me.  -PKR
%mark1 = ismember([colIdx rowIdx],[(1:length(x))' x],'rows','legacy');
% using MATLAB's built in C function for ismember - PKR
% https://www.mathworks.com/matlabcentral/answers/362036-search-for-elements-of-a-vector-in-a-matrix-without-using-ismember
mark1 = find(builtin('_ismemberhelper', colIdx, (1:length(x))') & ...
     builtin('_ismemberhelper', rowIdx, x));

Assignment_cost = (val(mark1));

